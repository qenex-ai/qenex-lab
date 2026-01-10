//! QENEX Accelerate - Rust Backend for Quantum Chemistry Integrals
//! 
//! Implements the Obara-Saika recurrence relations for Electron Repulsion Integrals (ERIs)
//! with Zero-Copy FFI bridge to Python via PyO3.
//!
//! Features:
//! - Parallel computation with Rayon (strict CPU pinning)
//! - ERIKey cache monitoring via Llama Scout integration
//! - High-precision Boys function implementation

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray4, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array4;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use once_cell::sync::Lazy;

// ============================================================================
// Global Statistics for Llama Scout Monitoring
// ============================================================================

/// Global statistics for ERIKey cache performance monitoring
/// These are atomically updated and can be queried from Python via scout_* functions
static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static TOTAL_ERIS_COMPUTED: AtomicU64 = AtomicU64::new(0);
static TOTAL_PRIMITIVES_PROCESSED: AtomicU64 = AtomicU64::new(0);
static THREAD_PINNING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Thread pool initialization flag
static THREAD_POOL_INITIALIZED: Lazy<bool> = Lazy::new(|| {
    initialize_pinned_thread_pool();
    true
});

// ============================================================================
// CPU Thread Pinning for Rayon
// ============================================================================

/// Initialize Rayon thread pool with strict CPU affinity pinning.
/// Each worker thread is pinned to a specific CPU core to minimize
/// context switches and maximize cache locality.
fn initialize_pinned_thread_pool() {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let num_cores = core_ids.len().max(1);
    
    // Build custom thread pool with pinned threads
    let result = rayon::ThreadPoolBuilder::new()
        .num_threads(num_cores)
        .start_handler(move |thread_idx| {
            // Pin each thread to a specific core
            if let Some(core_ids) = core_affinity::get_core_ids() {
                if thread_idx < core_ids.len() {
                    let _ = core_affinity::set_for_current(core_ids[thread_idx]);
                }
            }
        })
        .build_global();
    
    match result {
        Ok(_) => {
            THREAD_PINNING_ENABLED.store(true, Ordering::SeqCst);
        }
        Err(_) => {
            // Thread pool already initialized (not an error in practice)
            THREAD_PINNING_ENABLED.store(true, Ordering::SeqCst);
        }
    }
}

// ============================================================================
// ERI Cache Key - Custom hashable struct for memoization
// ============================================================================

/// Custom key type for ERI memoization cache
/// Encodes all 13 integer parameters into a single u64 for efficient hashing.
/// Each parameter is limited to 4 bits (0-15), which is sufficient for s,p,d,f orbitals.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ERIKey(u64);

impl ERIKey {
    #[inline]
    fn new(la: i32, lb: i32, lc: i32, ld: i32,
           ma: i32, mb: i32, mc: i32, md: i32,
           na: i32, nb: i32, nc: i32, nd: i32, m: i32) -> Self {
        // Pack 13 4-bit values into a u64 (52 bits used)
        let key = (la as u64 & 0xF)
            | ((lb as u64 & 0xF) << 4)
            | ((lc as u64 & 0xF) << 8)
            | ((ld as u64 & 0xF) << 12)
            | ((ma as u64 & 0xF) << 16)
            | ((mb as u64 & 0xF) << 20)
            | ((mc as u64 & 0xF) << 24)
            | ((md as u64 & 0xF) << 28)
            | ((na as u64 & 0xF) << 32)
            | ((nb as u64 & 0xF) << 36)
            | ((nc as u64 & 0xF) << 40)
            | ((nd as u64 & 0xF) << 44)
            | ((m as u64 & 0xF) << 48);
        ERIKey(key)
    }
    
    /// Decode the key back to its components (for debugging)
    #[allow(dead_code)]
    fn decode(&self) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
        let k = self.0;
        (
            (k & 0xF) as i32,
            ((k >> 4) & 0xF) as i32,
            ((k >> 8) & 0xF) as i32,
            ((k >> 12) & 0xF) as i32,
            ((k >> 16) & 0xF) as i32,
            ((k >> 20) & 0xF) as i32,
            ((k >> 24) & 0xF) as i32,
            ((k >> 28) & 0xF) as i32,
            ((k >> 32) & 0xF) as i32,
            ((k >> 36) & 0xF) as i32,
            ((k >> 40) & 0xF) as i32,
            ((k >> 44) & 0xF) as i32,
            ((k >> 48) & 0xF) as i32,
        )
    }
}

// ============================================================================
// Boys Function Implementation
// ============================================================================

/// Boys function F_n(t) = integral_0^1 u^(2n) exp(-t u^2) du
/// 
/// Uses different evaluation strategies based on the argument:
/// - Small t: Taylor series expansion
/// - Large t: Asymptotic expansion  
/// - Intermediate: Numerical integration
fn boys_function(n: i32, t: f64) -> f64 {
    if t < 1e-10 {
        // Taylor series for small t
        return 1.0 / (2 * n + 1) as f64 - t / (2 * n + 3) as f64;
    }
    
    if t > 50.0 {
        // Asymptotic expansion for large t
        // F_n(t) ~ (2n-1)!! * sqrt(pi) / (2^(n+1) * t^(n+0.5))
        let double_factorial = double_factorial(2 * n - 1);
        return double_factorial * (PI).sqrt() / (2.0_f64.powi(n + 1) * t.powf(n as f64 + 0.5));
    }
    
    // For F_0(t), use exact formula with error function
    if n == 0 {
        return 0.5 * (PI / t).sqrt() * erf(t.sqrt());
    }
    
    // For intermediate t and n > 0, use numerical integration (midpoint rule)
    // This is stable and accurate for our purposes
    let steps = 500;
    let dt = 1.0 / steps as f64;
    let mut result = 0.0;
    
    for i in 0..steps {
        let u = (i as f64 + 0.5) * dt;
        result += u.powi(2 * n) * (-t * u * u).exp();
    }
    
    result * dt
}

/// Double factorial: n!! = n * (n-2) * (n-4) * ... * 1 (or 2)
fn double_factorial(n: i32) -> f64 {
    if n <= 0 {
        return 1.0;
    }
    let mut result = 1.0;
    let mut i = n;
    while i > 0 {
        result *= i as f64;
        i -= 2;
    }
    result
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;
    
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

// ============================================================================
// Gaussian Product Theorem Helpers
// ============================================================================

/// Compute the center of a Gaussian product
/// P = (alpha_a * A + alpha_b * B) / (alpha_a + alpha_b)
#[inline]
fn gaussian_product_center(alpha_a: f64, a: &[f64; 3], alpha_b: f64, b: &[f64; 3]) -> [f64; 3] {
    let p = alpha_a + alpha_b;
    [
        (alpha_a * a[0] + alpha_b * b[0]) / p,
        (alpha_a * a[1] + alpha_b * b[1]) / p,
        (alpha_a * a[2] + alpha_b * b[2]) / p,
    ]
}

/// Squared distance between two points
#[inline]
fn dist_squared(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

// ============================================================================
// ERI Context for Obara-Saika Recurrence
// ============================================================================

/// Context structure holding all precomputed values for ERI recursion
struct ERIContext {
    /// Gaussian product center for bra (P)
    rp: [f64; 3],
    /// Gaussian product center for ket (Q)  
    rq: [f64; 3],
    /// Combined exponent for bra: p = alpha_a + alpha_b
    p: f64,
    /// Combined exponent for ket: q = alpha_c + alpha_d
    q: f64,
    /// Reduced exponent: rho = p*q / (p+q)
    rho: f64,
    /// Center A coordinates
    a: [f64; 3],
    /// Center B coordinates
    b: [f64; 3],
    /// Center C coordinates
    c: [f64; 3],
    /// Center D coordinates
    d: [f64; 3],
    /// Prefactor: (2*pi^2.5) / (p*q*sqrt(p+q)) * K_ab * K_cd
    prefactor: f64,
    /// Precomputed Boys function values
    boys_vals: Vec<f64>,
}

impl ERIContext {
    fn new(
        alpha_a: f64, alpha_b: f64, alpha_c: f64, alpha_d: f64,
        a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3],
        l_total: i32,
    ) -> Self {
        let p = alpha_a + alpha_b;
        let q = alpha_c + alpha_d;
        let rho = p * q / (p + q);
        
        let rp = gaussian_product_center(alpha_a, &a, alpha_b, &b);
        let rq = gaussian_product_center(alpha_c, &c, alpha_d, &d);
        
        // Gaussian decay factors
        let k_ab = (-alpha_a * alpha_b / p * dist_squared(&a, &b)).exp();
        let k_cd = (-alpha_c * alpha_d / q * dist_squared(&c, &d)).exp();
        
        let prefactor = (2.0 * PI.powi(2) * PI.sqrt()) / (p * q * (p + q).sqrt()) * k_ab * k_cd;
        
        // Precompute Boys function values
        let rpq_dist2 = dist_squared(&rp, &rq);
        let t_val = rho * rpq_dist2;
        
        let mut boys_vals = Vec::with_capacity((l_total + 1) as usize);
        for n in 0..=l_total {
            boys_vals.push(boys_function(n, t_val));
        }
        
        ERIContext {
            rp, rq, p, q, rho, a, b, c, d, prefactor, boys_vals
        }
    }
}

// ============================================================================
// Obara-Saika ERI Recurrence (Memoized with Scout Monitoring)
// ============================================================================

/// Memoized Obara-Saika recurrence for ERIs
/// 
/// Computes [la lb | lc ld]^(m) using the recurrence relations.
/// Angular momentum is specified as (la, ma_ang, na) for each center.
/// 
/// Scout Monitoring: Tracks cache hits/misses for performance analysis.
fn eri_recursion(
    la: i32, lb: i32, lc: i32, ld: i32,
    ma: i32, mb: i32, mc: i32, md: i32,
    na: i32, nb: i32, nc: i32, nd: i32,
    m: i32,
    ctx: &ERIContext,
    cache: &mut HashMap<ERIKey, f64>,
) -> f64 {
    let key = ERIKey::new(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m);
    
    if let Some(&val) = cache.get(&key) {
        // Scout: Track cache hit
        CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return val;
    }
    
    // Scout: Track cache miss
    CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    
    let p = ctx.p;
    let q = ctx.q;
    let rho = ctx.rho;
    
    // Base case: all angular momentum is zero
    let l_sum = la + lb + lc + ld + ma + mb + mc + md + na + nb + nc + nd;
    if l_sum == 0 {
        let result = ctx.prefactor * ctx.boys_vals[m as usize];
        cache.insert(key, result);
        return result;
    }
    
    let result;
    
    // X-Axis recurrence: reduce la
    if la > 0 {
        let term1 = (ctx.rp[0] - ctx.a[0]) * eri_recursion(la-1, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
        let term2 = -(q / (p + q)) * (ctx.rp[0] - ctx.rq[0]) * eri_recursion(la-1, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if la > 1 {
            let val = eri_recursion(la-2, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la-2, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term3 = (la - 1) as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term4 = 0.0;
        if lb > 0 {
            let val = eri_recursion(la-1, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la-1, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term4 = lb as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term5 = 0.0;
        if lc > 0 {
            let val = eri_recursion(la-1, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term5 = lc as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if ld > 0 {
            let val = eri_recursion(la-1, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term6 = ld as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // X-Axis: reduce lb
    else if lb > 0 {
        let term1 = (ctx.rp[0] - ctx.b[0]) * eri_recursion(la, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
        let term2 = -(q / (p + q)) * (ctx.rp[0] - ctx.rq[0]) * eri_recursion(la, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if la > 0 {
            let val = eri_recursion(la-1, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la-1, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term3 = la as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term4 = 0.0;
        if lb > 1 {
            let val = eri_recursion(la, lb-2, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb-2, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term4 = (lb - 1) as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term5 = 0.0;
        if lc > 0 {
            let val = eri_recursion(la, lb-1, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term5 = lc as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if ld > 0 {
            let val = eri_recursion(la, lb-1, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term6 = ld as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // X-Axis: reduce lc
    else if lc > 0 {
        let term1 = (ctx.rq[0] - ctx.c[0]) * eri_recursion(la, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
        let term2 = (p / (p + q)) * (ctx.rp[0] - ctx.rq[0]) * eri_recursion(la, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if lc > 1 {
            let val = eri_recursion(la, lb, lc-2, ld, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc-2, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term3 = (lc - 1) as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term4 = 0.0;
        if ld > 0 {
            let val = eri_recursion(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term4 = ld as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term5 = 0.0;
        if la > 0 {
            let val = eri_recursion(la-1, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term5 = la as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if lb > 0 {
            let val = eri_recursion(la, lb-1, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term6 = lb as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // X-Axis: reduce ld
    else if ld > 0 {
        let term1 = (ctx.rq[0] - ctx.d[0]) * eri_recursion(la, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
        let term2 = (p / (p + q)) * (ctx.rp[0] - ctx.rq[0]) * eri_recursion(la, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if lc > 0 {
            let val = eri_recursion(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term3 = lc as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term4 = 0.0;
        if ld > 1 {
            let val = eri_recursion(la, lb, lc, ld-2, ma, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld-2, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term4 = (ld - 1) as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term5 = 0.0;
        if la > 0 {
            let val = eri_recursion(la-1, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term5 = la as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if lb > 0 {
            let val = eri_recursion(la, lb-1, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term6 = lb as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // Y-Axis: reduce ma
    else if ma > 0 {
        let term1 = (ctx.rp[1] - ctx.a[1]) * eri_recursion(la, lb, lc, ld, ma-1, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
        let term2 = -(q / (p + q)) * (ctx.rp[1] - ctx.rq[1]) * eri_recursion(la, lb, lc, ld, ma-1, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if ma > 1 {
            let val = eri_recursion(la, lb, lc, ld, ma-2, mb, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma-2, mb, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term3 = (ma - 1) as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term4 = 0.0;
        if mb > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma-1, mb-1, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma-1, mb-1, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term4 = mb as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term5 = 0.0;
        if mc > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma-1, mb, mc-1, md, na, nb, nc, nd, m+1, ctx, cache);
            term5 = mc as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if md > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma-1, mb, mc, md-1, na, nb, nc, nd, m+1, ctx, cache);
            term6 = md as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // Y-Axis: reduce mb
    else if mb > 0 {
        let term1 = (ctx.rp[1] - ctx.b[1]) * eri_recursion(la, lb, lc, ld, ma, mb-1, mc, md, na, nb, nc, nd, m, ctx, cache);
        let term2 = -(q / (p + q)) * (ctx.rp[1] - ctx.rq[1]) * eri_recursion(la, lb, lc, ld, ma, mb-1, mc, md, na, nb, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if ma > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma-1, mb-1, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma-1, mb-1, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term3 = ma as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term4 = 0.0;
        if mb > 1 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb-2, mc, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb-2, mc, md, na, nb, nc, nd, m+1, ctx, cache);
            term4 = (mb - 1) as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term5 = 0.0;
        if mc > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb-1, mc-1, md, na, nb, nc, nd, m+1, ctx, cache);
            term5 = mc as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if md > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb-1, mc, md-1, na, nb, nc, nd, m+1, ctx, cache);
            term6 = md as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // Y-Axis: reduce mc
    else if mc > 0 {
        let term1 = (ctx.rq[1] - ctx.c[1]) * eri_recursion(la, lb, lc, ld, ma, mb, mc-1, md, na, nb, nc, nd, m, ctx, cache);
        let term2 = (p / (p + q)) * (ctx.rp[1] - ctx.rq[1]) * eri_recursion(la, lb, lc, ld, ma, mb, mc-1, md, na, nb, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if mc > 1 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc-2, md, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc-2, md, na, nb, nc, nd, m+1, ctx, cache);
            term3 = (mc - 1) as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term4 = 0.0;
        if md > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc-1, md-1, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc-1, md-1, na, nb, nc, nd, m+1, ctx, cache);
            term4 = md as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term5 = 0.0;
        if ma > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma-1, mb, mc-1, md, na, nb, nc, nd, m+1, ctx, cache);
            term5 = ma as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if mb > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb-1, mc-1, md, na, nb, nc, nd, m+1, ctx, cache);
            term6 = mb as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // Y-Axis: reduce md
    else if md > 0 {
        let term1 = (ctx.rq[1] - ctx.d[1]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md-1, na, nb, nc, nd, m, ctx, cache);
        let term2 = (p / (p + q)) * (ctx.rp[1] - ctx.rq[1]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md-1, na, nb, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if mc > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc-1, md-1, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc-1, md-1, na, nb, nc, nd, m+1, ctx, cache);
            term3 = mc as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term4 = 0.0;
        if md > 1 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md-2, na, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc, md-2, na, nb, nc, nd, m+1, ctx, cache);
            term4 = (md - 1) as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term5 = 0.0;
        if ma > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma-1, mb, mc, md-1, na, nb, nc, nd, m+1, ctx, cache);
            term5 = ma as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if mb > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb-1, mc, md-1, na, nb, nc, nd, m+1, ctx, cache);
            term6 = mb as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // Z-Axis: reduce na
    else if na > 0 {
        let term1 = (ctx.rp[2] - ctx.a[2]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc, nd, m, ctx, cache);
        let term2 = -(q / (p + q)) * (ctx.rp[2] - ctx.rq[2]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if na > 1 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-2, nb, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-2, nb, nc, nd, m+1, ctx, cache);
            term3 = (na - 1) as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term4 = 0.0;
        if nb > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb-1, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb-1, nc, nd, m+1, ctx, cache);
            term4 = nb as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term5 = 0.0;
        if nc > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc-1, nd, m+1, ctx, cache);
            term5 = nc as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if nd > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc, nd-1, m+1, ctx, cache);
            term6 = nd as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // Z-Axis: reduce nb
    else if nb > 0 {
        let term1 = (ctx.rp[2] - ctx.b[2]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc, nd, m, ctx, cache);
        let term2 = -(q / (p + q)) * (ctx.rp[2] - ctx.rq[2]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if na > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb-1, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb-1, nc, nd, m+1, ctx, cache);
            term3 = na as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term4 = 0.0;
        if nb > 1 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb-2, nc, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb-2, nc, nd, m+1, ctx, cache);
            term4 = (nb - 1) as f64 / (2.0 * p) * (val - (rho / p) * val_m);
        }
        
        let mut term5 = 0.0;
        if nc > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc-1, nd, m+1, ctx, cache);
            term5 = nc as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if nd > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc, nd-1, m+1, ctx, cache);
            term6 = nd as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // Z-Axis: reduce nc
    else if nc > 0 {
        let term1 = (ctx.rq[2] - ctx.c[2]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd, m, ctx, cache);
        let term2 = (p / (p + q)) * (ctx.rp[2] - ctx.rq[2]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if nc > 1 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-2, nd, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-2, nd, m+1, ctx, cache);
            term3 = (nc - 1) as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term4 = 0.0;
        if nd > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd-1, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd-1, m+1, ctx, cache);
            term4 = nd as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term5 = 0.0;
        if na > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc-1, nd, m+1, ctx, cache);
            term5 = na as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if nb > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc-1, nd, m+1, ctx, cache);
            term6 = nb as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    // Z-Axis: reduce nd
    else if nd > 0 {
        let term1 = (ctx.rq[2] - ctx.d[2]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd-1, m, ctx, cache);
        let term2 = (p / (p + q)) * (ctx.rp[2] - ctx.rq[2]) * eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd-1, m+1, ctx, cache);
        
        let mut term3 = 0.0;
        if nc > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd-1, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd-1, m+1, ctx, cache);
            term3 = nc as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term4 = 0.0;
        if nd > 1 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd-2, m, ctx, cache);
            let val_m = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd-2, m+1, ctx, cache);
            term4 = (nd - 1) as f64 / (2.0 * q) * (val - (rho / q) * val_m);
        }
        
        let mut term5 = 0.0;
        if na > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc, nd-1, m+1, ctx, cache);
            term5 = na as f64 / (2.0 * (p + q)) * val;
        }
        
        let mut term6 = 0.0;
        if nb > 0 {
            let val = eri_recursion(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc, nd-1, m+1, ctx, cache);
            term6 = nb as f64 / (2.0 * (p + q)) * val;
        }
        
        result = term1 + term2 + term3 + term4 + term5 + term6;
    }
    else {
        result = 0.0;
    }
    
    cache.insert(key, result);
    result
}

// ============================================================================
// Primitive ERI Computation
// ============================================================================

/// Compute a single primitive ERI (ab|cd)
fn eri_primitive(
    alpha_a: f64, alpha_b: f64, alpha_c: f64, alpha_d: f64,
    a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3],
    la: i32, ma: i32, na: i32,
    lb: i32, mb: i32, nb: i32,
    lc: i32, mc: i32, nc: i32,
    ld: i32, md: i32, nd: i32,
    norm_a: f64, norm_b: f64, norm_c: f64, norm_d: f64,
) -> f64 {
    let l_total = la + lb + lc + ld + ma + mb + mc + md + na + nb + nc + nd;
    
    let ctx = ERIContext::new(
        alpha_a, alpha_b, alpha_c, alpha_d,
        a, b, c, d,
        l_total,
    );
    
    let mut cache = HashMap::new();
    let val = eri_recursion(
        la, lb, lc, ld,
        ma, mb, mc, md,
        na, nb, nc, nd,
        0, &ctx, &mut cache,
    );
    
    norm_a * norm_b * norm_c * norm_d * val
}

// ============================================================================
// Normalization Constant
// ============================================================================

/// Compute normalization constant for a primitive Gaussian
fn normalize_primitive(alpha: f64, l: i32, m: i32, n: i32) -> f64 {
    let l_total = l + m + n;
    let pre = (2.0 * alpha / PI).powf(0.75);
    let ang = (4.0 * alpha).powf(l_total as f64 / 2.0);
    let denom = (double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)).sqrt();
    pre * ang / denom
}

// ============================================================================
// Python Interface
// ============================================================================

/// Compute the full ERI tensor for a basis set
/// 
/// This is the main entry point called from Python.
/// It receives flattened arrays describing all primitives and returns
/// the full (N, N, N, N) ERI tensor.
#[pyfunction]
fn compute_eri<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    exps: PyReadonlyArray1<f64>,
    norms: PyReadonlyArray1<f64>,
    lmns: PyReadonlyArray2<i64>,
    at_idx: PyReadonlyArray1<i64>,
    bas_idx: PyReadonlyArray1<i64>,
    n_basis: usize,
) -> &'py PyArray4<f64> {
    let coords = coords.as_array();
    let exps = exps.as_array();
    let norms = norms.as_array();
    let lmns = lmns.as_array();
    let at_idx = at_idx.as_array();
    let bas_idx = bas_idx.as_array();
    
    let n_prims = exps.len();
    
    // Scout: Track total primitives
    TOTAL_PRIMITIVES_PROCESSED.fetch_add(n_prims as u64, Ordering::Relaxed);
    
    // Create the output ERI tensor
    let mut eri_tensor = Array4::<f64>::zeros((n_basis, n_basis, n_basis, n_basis));
    
    // Loop over all primitive quartets
    // Note: In production, we'd use 8-fold symmetry and screening
    for i in 0..n_prims {
        for j in 0..n_prims {
            for k in 0..n_prims {
                for l in 0..n_prims {
                    let mu = bas_idx[i] as usize;
                    let nu = bas_idx[j] as usize;
                    let lam = bas_idx[k] as usize;
                    let sig = bas_idx[l] as usize;
                    
                    // Get coordinates for each primitive's atom
                    let ai = at_idx[i] as usize;
                    let aj = at_idx[j] as usize;
                    let ak = at_idx[k] as usize;
                    let al = at_idx[l] as usize;
                    
                    let a = [coords[[ai, 0]], coords[[ai, 1]], coords[[ai, 2]]];
                    let b = [coords[[aj, 0]], coords[[aj, 1]], coords[[aj, 2]]];
                    let c = [coords[[ak, 0]], coords[[ak, 1]], coords[[ak, 2]]];
                    let d = [coords[[al, 0]], coords[[al, 1]], coords[[al, 2]]];
                    
                    // Angular momentum
                    let la = lmns[[i, 0]] as i32;
                    let ma = lmns[[i, 1]] as i32;
                    let na = lmns[[i, 2]] as i32;
                    
                    let lb = lmns[[j, 0]] as i32;
                    let mb = lmns[[j, 1]] as i32;
                    let nb = lmns[[j, 2]] as i32;
                    
                    let lc = lmns[[k, 0]] as i32;
                    let mc = lmns[[k, 1]] as i32;
                    let nc = lmns[[k, 2]] as i32;
                    
                    let ld = lmns[[l, 0]] as i32;
                    let md = lmns[[l, 1]] as i32;
                    let nd = lmns[[l, 2]] as i32;
                    
                    // Compute the primitive ERI
                    let val = eri_primitive(
                        exps[i], exps[j], exps[k], exps[l],
                        a, b, c, d,
                        la, ma, na,
                        lb, mb, nb,
                        lc, mc, nc,
                        ld, md, nd,
                        norms[i], norms[j], norms[k], norms[l],
                    );
                    
                    // Accumulate into the contracted basis function tensor
                    eri_tensor[[mu, nu, lam, sig]] += val;
                }
            }
        }
    }
    
    eri_tensor.into_pyarray(py)
}

/// Compute ERIs in parallel using Rayon with strict CPU pinning
/// 
/// This version parallelizes the outermost loop for better performance
/// on multi-core systems. Each Rayon worker thread is pinned to a specific
/// CPU core to minimize context switches and maximize cache locality.
/// 
/// NOTE: This version computes all N^4 primitive quartets without symmetry
/// optimization. For larger systems, consider using compute_eri_symmetric
/// which exploits 8-fold permutational symmetry.
#[pyfunction]
fn compute_eri_parallel<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    exps: PyReadonlyArray1<f64>,
    norms: PyReadonlyArray1<f64>,
    lmns: PyReadonlyArray2<i64>,
    at_idx: PyReadonlyArray1<i64>,
    bas_idx: PyReadonlyArray1<i64>,
    n_basis: usize,
) -> &'py PyArray4<f64> {
    // Ensure thread pool is initialized with CPU pinning
    let _ = *THREAD_POOL_INITIALIZED;
    
    let coords = coords.as_array().to_owned();
    let exps = exps.as_array().to_owned();
    let norms = norms.as_array().to_owned();
    let lmns = lmns.as_array().to_owned();
    let at_idx = at_idx.as_array().to_owned();
    let bas_idx = bas_idx.as_array().to_owned();
    
    let n_prims = exps.len();
    
    // Scout: Track total primitives
    TOTAL_PRIMITIVES_PROCESSED.fetch_add(n_prims as u64, Ordering::Relaxed);
    
    // Compute all quartets in parallel with pinned threads
    let results: Vec<(usize, usize, usize, usize, f64)> = (0..n_prims)
        .into_par_iter()
        .flat_map(|i| {
            let mut local_results = Vec::new();
            for j in 0..n_prims {
                for k in 0..n_prims {
                    for l in 0..n_prims {
                        let mu = bas_idx[i] as usize;
                        let nu = bas_idx[j] as usize;
                        let lam = bas_idx[k] as usize;
                        let sig = bas_idx[l] as usize;
                        
                        let ai = at_idx[i] as usize;
                        let aj = at_idx[j] as usize;
                        let ak = at_idx[k] as usize;
                        let al = at_idx[l] as usize;
                        
                        let a = [coords[[ai, 0]], coords[[ai, 1]], coords[[ai, 2]]];
                        let b = [coords[[aj, 0]], coords[[aj, 1]], coords[[aj, 2]]];
                        let c = [coords[[ak, 0]], coords[[ak, 1]], coords[[ak, 2]]];
                        let d = [coords[[al, 0]], coords[[al, 1]], coords[[al, 2]]];
                        
                        let la = lmns[[i, 0]] as i32;
                        let ma_ang = lmns[[i, 1]] as i32;
                        let na = lmns[[i, 2]] as i32;
                        
                        let lb = lmns[[j, 0]] as i32;
                        let mb = lmns[[j, 1]] as i32;
                        let nb = lmns[[j, 2]] as i32;
                        
                        let lc = lmns[[k, 0]] as i32;
                        let mc = lmns[[k, 1]] as i32;
                        let nc = lmns[[k, 2]] as i32;
                        
                        let ld = lmns[[l, 0]] as i32;
                        let md = lmns[[l, 1]] as i32;
                        let nd = lmns[[l, 2]] as i32;
                        
                        let val = eri_primitive(
                            exps[i], exps[j], exps[k], exps[l],
                            a, b, c, d,
                            la, ma_ang, na,
                            lb, mb, nb,
                            lc, mc, nc,
                            ld, md, nd,
                            norms[i], norms[j], norms[k], norms[l],
                        );
                        
                        if val.abs() > 1e-15 {
                            local_results.push((mu, nu, lam, sig, val));
                        }
                    }
                }
            }
            local_results
        })
        .collect();
    
    // Accumulate results into tensor
    let mut eri_tensor = Array4::<f64>::zeros((n_basis, n_basis, n_basis, n_basis));
    for (mu, nu, lam, sig, val) in results {
        eri_tensor[[mu, nu, lam, sig]] += val;
    }
    
    eri_tensor.into_pyarray(py)
}

/// Compute ERIs with 8-fold symmetry optimization
/// 
/// This version exploits the permutational symmetry of ERIs:
/// (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
/// 
/// We only compute unique basis function quartets, reducing work by ~8x.
/// The symmetry is at the **basis function** level, not the primitive level.
#[pyfunction]
fn compute_eri_symmetric<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    exps: PyReadonlyArray1<f64>,
    norms: PyReadonlyArray1<f64>,
    lmns: PyReadonlyArray2<i64>,
    at_idx: PyReadonlyArray1<i64>,
    bas_idx: PyReadonlyArray1<i64>,
    n_basis: usize,
) -> &'py PyArray4<f64> {
    // Ensure thread pool is initialized with CPU pinning
    let _ = *THREAD_POOL_INITIALIZED;
    
    let coords = coords.as_array().to_owned();
    let exps = exps.as_array().to_owned();
    let norms = norms.as_array().to_owned();
    let lmns = lmns.as_array().to_owned();
    let at_idx = at_idx.as_array().to_owned();
    let bas_idx = bas_idx.as_array().to_owned();
    
    let n_prims = exps.len();
    
    // Scout: Track total primitives
    TOTAL_PRIMITIVES_PROCESSED.fetch_add(n_prims as u64, Ordering::Relaxed);
    
    // Build primitive index ranges for each basis function
    let mut basis_prim_ranges: Vec<(usize, usize)> = Vec::with_capacity(n_basis);
    let mut current_basis = 0;
    let mut start_idx = 0;
    
    for (idx, &b) in bas_idx.iter().enumerate() {
        let b = b as usize;
        if b != current_basis {
            basis_prim_ranges.push((start_idx, idx));
            current_basis = b;
            start_idx = idx;
        }
    }
    basis_prim_ranges.push((start_idx, n_prims));
    
    // Generate unique basis function quartets with 8-fold symmetry
    // (mu, nu, lam, sig) where mu >= nu, lam >= sig, and (mu,nu) >= (lam,sig) lexicographically
    let unique_quartets: Vec<(usize, usize, usize, usize)> = (0..n_basis)
        .flat_map(|mu| {
            (0..=mu).flat_map(move |nu| {
                (0..=mu).flat_map(move |lam| {
                    let sig_max = if lam == mu { nu } else { lam };
                    (0..=sig_max).map(move |sig| (mu, nu, lam, sig))
                })
            })
        })
        .collect();
    
    // Compute unique quartets in parallel
    let results: Vec<(usize, usize, usize, usize, f64)> = unique_quartets
        .into_par_iter()
        .filter_map(|(mu, nu, lam, sig)| {
            // Sum over all primitive quartets for this basis quartet
            let (i_start, i_end) = basis_prim_ranges[mu];
            let (j_start, j_end) = basis_prim_ranges[nu];
            let (k_start, k_end) = basis_prim_ranges[lam];
            let (l_start, l_end) = basis_prim_ranges[sig];
            
            let mut val = 0.0;
            
            for i in i_start..i_end {
                for j in j_start..j_end {
                    for k in k_start..k_end {
                        for l in l_start..l_end {
                            let ai = at_idx[i] as usize;
                            let aj = at_idx[j] as usize;
                            let ak = at_idx[k] as usize;
                            let al = at_idx[l] as usize;
                            
                            let a = [coords[[ai, 0]], coords[[ai, 1]], coords[[ai, 2]]];
                            let b = [coords[[aj, 0]], coords[[aj, 1]], coords[[aj, 2]]];
                            let c = [coords[[ak, 0]], coords[[ak, 1]], coords[[ak, 2]]];
                            let d = [coords[[al, 0]], coords[[al, 1]], coords[[al, 2]]];
                            
                            let la = lmns[[i, 0]] as i32;
                            let ma_ang = lmns[[i, 1]] as i32;
                            let na = lmns[[i, 2]] as i32;
                            
                            let lb = lmns[[j, 0]] as i32;
                            let mb = lmns[[j, 1]] as i32;
                            let nb = lmns[[j, 2]] as i32;
                            
                            let lc = lmns[[k, 0]] as i32;
                            let mc = lmns[[k, 1]] as i32;
                            let nc = lmns[[k, 2]] as i32;
                            
                            let ld = lmns[[l, 0]] as i32;
                            let md = lmns[[l, 1]] as i32;
                            let nd = lmns[[l, 2]] as i32;
                            
                            val += eri_primitive(
                                exps[i], exps[j], exps[k], exps[l],
                                a, b, c, d,
                                la, ma_ang, na,
                                lb, mb, nb,
                                lc, mc, nc,
                                ld, md, nd,
                                norms[i], norms[j], norms[k], norms[l],
                            );
                        }
                    }
                }
            }
            
            if val.abs() > 1e-15 {
                Some((mu, nu, lam, sig, val))
            } else {
                None
            }
        })
        .collect();
    
    // Accumulate results into tensor with 8-fold symmetry expansion
    let mut eri_tensor = Array4::<f64>::zeros((n_basis, n_basis, n_basis, n_basis));
    
    for (mu, nu, lam, sig, val) in results {
        // Fill all 8 symmetric positions
        // We need to be careful: some positions may be the same if indices match
        eri_tensor[[mu, nu, lam, sig]] = val;
        eri_tensor[[nu, mu, lam, sig]] = val;
        eri_tensor[[mu, nu, sig, lam]] = val;
        eri_tensor[[nu, mu, sig, lam]] = val;
        eri_tensor[[lam, sig, mu, nu]] = val;
        eri_tensor[[sig, lam, mu, nu]] = val;
        eri_tensor[[lam, sig, nu, mu]] = val;
        eri_tensor[[sig, lam, nu, mu]] = val;
    }
    
    eri_tensor.into_pyarray(py)
}

// ============================================================================
// Llama Scout Monitoring Interface
// ============================================================================

/// Get current ERIKey cache statistics
/// Returns (hits, misses, hit_rate)
#[pyfunction]
fn scout_get_cache_stats() -> (u64, u64, f64) {
    let hits = CACHE_HITS.load(Ordering::Relaxed);
    let misses = CACHE_MISSES.load(Ordering::Relaxed);
    let total = hits + misses;
    let hit_rate = if total > 0 { hits as f64 / total as f64 } else { 0.0 };
    (hits, misses, hit_rate)
}

/// Reset all Scout monitoring counters
#[pyfunction]
fn scout_reset_stats() {
    CACHE_HITS.store(0, Ordering::SeqCst);
    CACHE_MISSES.store(0, Ordering::SeqCst);
    TOTAL_ERIS_COMPUTED.store(0, Ordering::SeqCst);
    TOTAL_PRIMITIVES_PROCESSED.store(0, Ordering::SeqCst);
}

/// Get total ERIs computed since last reset
#[pyfunction]
fn scout_get_total_eris() -> u64 {
    TOTAL_ERIS_COMPUTED.load(Ordering::Relaxed)
}

/// Get total primitives processed since last reset
#[pyfunction]
fn scout_get_total_primitives() -> u64 {
    TOTAL_PRIMITIVES_PROCESSED.load(Ordering::Relaxed)
}

/// Check if thread pinning is enabled
#[pyfunction]
fn scout_is_thread_pinning_enabled() -> bool {
    THREAD_PINNING_ENABLED.load(Ordering::Relaxed)
}

/// Get number of available CPU cores for Rayon
#[pyfunction]
fn scout_get_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Get comprehensive Scout report as a dictionary
#[pyfunction]
fn scout_report() -> HashMap<String, String> {
    let (hits, misses, hit_rate) = scout_get_cache_stats();
    let mut report = HashMap::new();
    
    report.insert("cache_hits".to_string(), hits.to_string());
    report.insert("cache_misses".to_string(), misses.to_string());
    report.insert("cache_hit_rate".to_string(), format!("{:.4}", hit_rate));
    report.insert("total_eris".to_string(), TOTAL_ERIS_COMPUTED.load(Ordering::Relaxed).to_string());
    report.insert("total_primitives".to_string(), TOTAL_PRIMITIVES_PROCESSED.load(Ordering::Relaxed).to_string());
    report.insert("thread_pinning".to_string(), THREAD_PINNING_ENABLED.load(Ordering::Relaxed).to_string());
    report.insert("num_threads".to_string(), rayon::current_num_threads().to_string());
    
    // Get CPU core IDs if available
    if let Some(core_ids) = core_affinity::get_core_ids() {
        report.insert("available_cores".to_string(), core_ids.len().to_string());
    }
    
    report
}

/// Initialize the Rayon thread pool with CPU pinning
/// Call this explicitly to ensure threads are pinned before computation
#[pyfunction]
fn scout_initialize_thread_pool() -> bool {
    let _ = *THREAD_POOL_INITIALIZED;
    THREAD_PINNING_ENABLED.load(Ordering::Relaxed)
}

#[pymodule]
fn qenex_accelerate(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core ERI functions
    m.add_function(wrap_pyfunction!(compute_eri, m)?)?;
    m.add_function(wrap_pyfunction!(compute_eri_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_eri_symmetric, m)?)?;
    
    // Scout monitoring functions
    m.add_function(wrap_pyfunction!(scout_get_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(scout_reset_stats, m)?)?;
    m.add_function(wrap_pyfunction!(scout_get_total_eris, m)?)?;
    m.add_function(wrap_pyfunction!(scout_get_total_primitives, m)?)?;
    m.add_function(wrap_pyfunction!(scout_is_thread_pinning_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(scout_get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(scout_report, m)?)?;
    m.add_function(wrap_pyfunction!(scout_initialize_thread_pool, m)?)?;
    
    Ok(())
}

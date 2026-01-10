"""
QENEX Neuroscience Module
=========================
Computational neuroscience covering:
- Single neuron dynamics (Hodgkin-Huxley, Izhikevich, LIF)
- Neural networks and population dynamics
- Synaptic plasticity (STDP, Hebbian)
- Brain connectivity and graph theory
- Neural coding and information theory
- Cognitive architectures

Physical Constants:
- Membrane capacitance C_m ~ 1 µF/cm²
- Resting potential V_rest ~ -65 mV
- Action potential threshold ~ -55 mV
- Synaptic time constants ~ 1-100 ms

References:
- Hodgkin & Huxley (1952) - Action potential model
- Izhikevich (2003) - Simple model of spiking neurons
- Dayan & Abbott (2001) - Theoretical Neuroscience
- Gerstner et al. (2014) - Neuronal Dynamics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Union
from enum import Enum, auto
from abc import ABC, abstractmethod
import json

# =============================================================================
# PHYSICAL CONSTANTS (Neurophysiology)
# =============================================================================

# Membrane properties
C_M = 1.0                  # Membrane capacitance [µF/cm²]
R_M = 10.0                 # Membrane resistance [kΩ·cm²]
TAU_M = C_M * R_M          # Membrane time constant [ms]

# Reversal potentials [mV]
E_NA = 50.0                # Sodium reversal
E_K = -77.0                # Potassium reversal
E_LEAK = -54.4             # Leak reversal
E_EXC = 0.0                # Excitatory synaptic reversal
E_INH = -80.0              # Inhibitory synaptic reversal

# Conductances [mS/cm²]
G_NA = 120.0               # Sodium conductance
G_K = 36.0                 # Potassium conductance
G_LEAK = 0.3               # Leak conductance

# Typical values
V_REST = -65.0             # Resting potential [mV]
V_THRESHOLD = -55.0        # Spike threshold [mV]
V_PEAK = 30.0              # Spike peak [mV]
V_RESET = -70.0            # Reset potential [mV]

# Synaptic time constants [ms]
TAU_AMPA = 2.0             # AMPA receptor
TAU_NMDA = 100.0           # NMDA receptor
TAU_GABA_A = 5.0           # GABA-A receptor
TAU_GABA_B = 100.0         # GABA-B receptor

# STDP parameters
TAU_STDP_PLUS = 20.0       # LTP time constant [ms]
TAU_STDP_MINUS = 20.0      # LTD time constant [ms]
A_PLUS = 0.01              # LTP amplitude
A_MINUS = 0.012            # LTD amplitude (slightly larger for stability)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class NeuronType(Enum):
    """Types of neurons."""
    EXCITATORY = auto()      # Glutamatergic
    INHIBITORY = auto()      # GABAergic
    MODULATORY = auto()      # Dopamine, serotonin, etc.
    SENSORY = auto()         # Input neurons
    MOTOR = auto()           # Output neurons


class SynapseType(Enum):
    """Types of synapses."""
    AMPA = auto()            # Fast excitatory
    NMDA = auto()            # Slow excitatory, voltage-dependent
    GABA_A = auto()          # Fast inhibitory
    GABA_B = auto()          # Slow inhibitory
    ELECTRICAL = auto()      # Gap junction


class PlasticityRule(Enum):
    """Synaptic plasticity rules."""
    NONE = auto()
    HEBBIAN = auto()         # Fire together, wire together
    STDP = auto()            # Spike-timing dependent plasticity
    BCM = auto()             # Bienenstock-Cooper-Munro
    HOMEOSTATIC = auto()     # Synaptic scaling


class BrainRegion(Enum):
    """Major brain regions."""
    CORTEX = auto()
    HIPPOCAMPUS = auto()
    THALAMUS = auto()
    BASAL_GANGLIA = auto()
    CEREBELLUM = auto()
    BRAINSTEM = auto()
    AMYGDALA = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Spike:
    """Single spike event.
    
    Attributes:
        neuron_id: ID of spiking neuron
        time: Spike time [ms]
    """
    neuron_id: int
    time: float


@dataclass
class SynapticConnection:
    """Synaptic connection between neurons.
    
    Attributes:
        pre_id: Presynaptic neuron ID
        post_id: Postsynaptic neuron ID
        weight: Synaptic weight
        delay: Axonal delay [ms]
        synapse_type: Type of synapse
    """
    pre_id: int
    post_id: int
    weight: float = 1.0
    delay: float = 1.0
    synapse_type: SynapseType = SynapseType.AMPA


@dataclass 
class NeuronState:
    """State of a single neuron.
    
    Attributes:
        V: Membrane potential [mV]
        spike: Whether neuron spiked this timestep
        last_spike: Time of last spike [ms]
    """
    V: float = V_REST
    spike: bool = False
    last_spike: float = -1000.0


# =============================================================================
# NEURON MODELS
# =============================================================================

class Neuron(ABC):
    """Abstract base class for neuron models."""
    
    @abstractmethod
    def step(self, I_ext: float, dt: float) -> bool:
        """Advance neuron one timestep.
        
        Args:
            I_ext: External input current [nA or µA/cm²]
            dt: Timestep [ms]
        
        Returns:
            True if neuron spiked
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset neuron to initial state."""
        pass


class LeakyIntegrateAndFire(Neuron):
    """Leaky Integrate-and-Fire neuron.
    
    The simplest spiking neuron model:
    τ_m dV/dt = -(V - V_rest) + R_m * I_ext
    
    When V > V_thresh: emit spike, reset to V_reset
    
    Example:
        >>> lif = LeakyIntegrateAndFire()
        >>> for t in range(1000):
        ...     spike = lif.step(I_ext=15.0, dt=0.1)
        ...     if spike:
        ...         print(f"Spike at t={t*0.1:.1f} ms")
    """
    
    def __init__(
        self,
        tau_m: float = TAU_M,
        V_rest: float = V_REST,
        V_thresh: float = V_THRESHOLD,
        V_reset: float = V_RESET,
        R_m: float = R_M,
        t_refract: float = 2.0
    ):
        """Initialize LIF neuron.
        
        Args:
            tau_m: Membrane time constant [ms]
            V_rest: Resting potential [mV]
            V_thresh: Spike threshold [mV]
            V_reset: Reset potential [mV]
            R_m: Membrane resistance [MΩ]
            t_refract: Refractory period [ms]
        """
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.R_m = R_m
        self.t_refract = t_refract
        
        self.V = V_rest
        self.t_since_spike = 1000.0  # Time since last spike
        
    def step(self, I_ext: float, dt: float) -> bool:
        """Advance neuron one timestep."""
        self.t_since_spike += dt
        
        # Refractory period
        if self.t_since_spike < self.t_refract:
            return False
        
        # Membrane dynamics
        dV = (-(self.V - self.V_rest) + self.R_m * I_ext) / self.tau_m
        self.V += dV * dt
        
        # Spike?
        if self.V >= self.V_thresh:
            self.V = self.V_reset
            self.t_since_spike = 0.0
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.V = self.V_rest
        self.t_since_spike = 1000.0
    
    def firing_rate(self, I_ext: float) -> float:
        """Calculate steady-state firing rate (f-I curve).
        
        Args:
            I_ext: Input current [nA]
        
        Returns:
            Firing rate [Hz]
        """
        if I_ext <= 0:
            return 0.0
        
        # Threshold current
        I_thresh = (self.V_thresh - self.V_rest) / self.R_m
        
        if I_ext <= I_thresh:
            return 0.0
        
        # Inter-spike interval
        V_inf = self.V_rest + self.R_m * I_ext
        isi = self.tau_m * np.log((V_inf - self.V_reset) / (V_inf - self.V_thresh))
        isi += self.t_refract
        
        return 1000.0 / isi  # Convert ms to Hz


class HodgkinHuxley(Neuron):
    """Hodgkin-Huxley neuron model.
    
    The biophysically detailed model of action potential generation:
    
    C_m dV/dt = -g_Na*m³*h*(V-E_Na) - g_K*n⁴*(V-E_K) - g_L*(V-E_L) + I_ext
    
    where m, h, n are gating variables with voltage-dependent dynamics.
    
    Nobel Prize in Physiology or Medicine 1963.
    
    Example:
        >>> hh = HodgkinHuxley()
        >>> V_trace = []
        >>> for t in range(10000):
        ...     spike = hh.step(I_ext=10.0, dt=0.01)
        ...     V_trace.append(hh.V)
    """
    
    def __init__(
        self,
        C_m: float = C_M,
        g_Na: float = G_NA,
        g_K: float = G_K,
        g_L: float = G_LEAK,
        E_Na: float = E_NA,
        E_K: float = E_K,
        E_L: float = E_LEAK
    ):
        """Initialize HH neuron.
        
        Args:
            C_m: Membrane capacitance [µF/cm²]
            g_Na: Sodium conductance [mS/cm²]
            g_K: Potassium conductance [mS/cm²]
            g_L: Leak conductance [mS/cm²]
            E_Na: Sodium reversal [mV]
            E_K: Potassium reversal [mV]
            E_L: Leak reversal [mV]
        """
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        
        # State variables
        self.V = -65.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        
        self._last_V = self.V
        
    def _alpha_m(self, V: float) -> float:
        """Rate constant for m activation."""
        if abs(V + 40) < 1e-6:
            return 1.0
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    
    def _beta_m(self, V: float) -> float:
        """Rate constant for m deactivation."""
        return 4.0 * np.exp(-(V + 65) / 18)
    
    def _alpha_h(self, V: float) -> float:
        """Rate constant for h activation."""
        return 0.07 * np.exp(-(V + 65) / 20)
    
    def _beta_h(self, V: float) -> float:
        """Rate constant for h deactivation."""
        return 1.0 / (1 + np.exp(-(V + 35) / 10))
    
    def _alpha_n(self, V: float) -> float:
        """Rate constant for n activation."""
        if abs(V + 55) < 1e-6:
            return 0.1
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    
    def _beta_n(self, V: float) -> float:
        """Rate constant for n deactivation."""
        return 0.125 * np.exp(-(V + 65) / 80)
    
    def step(self, I_ext: float, dt: float) -> bool:
        """Advance neuron one timestep using RK4."""
        self._last_V = self.V
        
        # Ionic currents
        I_Na = self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)
        I_K = self.g_K * self.n**4 * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)
        
        # Membrane equation
        dV = (I_ext - I_Na - I_K - I_L) / self.C_m
        
        # Gating variable dynamics
        dm = self._alpha_m(self.V) * (1 - self.m) - self._beta_m(self.V) * self.m
        dh = self._alpha_h(self.V) * (1 - self.h) - self._beta_h(self.V) * self.h
        dn = self._alpha_n(self.V) * (1 - self.n) - self._beta_n(self.V) * self.n
        
        # Euler update (for simplicity; RK4 is better)
        self.V += dV * dt
        self.m += dm * dt
        self.h += dh * dt
        self.n += dn * dt
        
        # Clamp gating variables
        self.m = np.clip(self.m, 0, 1)
        self.h = np.clip(self.h, 0, 1)
        self.n = np.clip(self.n, 0, 1)
        
        # Spike detection (threshold crossing)
        return self._last_V < 0 and self.V >= 0
    
    def reset(self) -> None:
        """Reset to resting state."""
        self.V = -65.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32


class IzhikevichNeuron(Neuron):
    """Izhikevich neuron model.
    
    Computationally efficient model that can reproduce diverse firing patterns:
    
    dv/dt = 0.04v² + 5v + 140 - u + I
    du/dt = a(bv - u)
    
    if v >= 30: v = c, u = u + d
    
    Parameters (a, b, c, d) determine neuron type:
    - Regular Spiking (RS): a=0.02, b=0.2, c=-65, d=8
    - Intrinsically Bursting (IB): a=0.02, b=0.2, c=-55, d=4
    - Chattering (CH): a=0.02, b=0.2, c=-50, d=2
    - Fast Spiking (FS): a=0.1, b=0.2, c=-65, d=2
    - Low-Threshold Spiking (LTS): a=0.02, b=0.25, c=-65, d=2
    
    Example:
        >>> izh = IzhikevichNeuron.regular_spiking()
        >>> spikes = []
        >>> for t in range(1000):
        ...     if izh.step(I_ext=10, dt=1.0):
        ...         spikes.append(t)
    """
    
    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0
    ):
        """Initialize Izhikevich neuron.
        
        Args:
            a: Recovery time scale
            b: Recovery sensitivity to voltage
            c: Post-spike reset voltage [mV]
            d: Post-spike recovery increment
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        self.v = c
        self.u = b * c
        
    @classmethod
    def regular_spiking(cls) -> 'IzhikevichNeuron':
        """Create regular spiking excitatory neuron."""
        return cls(a=0.02, b=0.2, c=-65, d=8)
    
    @classmethod
    def intrinsically_bursting(cls) -> 'IzhikevichNeuron':
        """Create intrinsically bursting neuron."""
        return cls(a=0.02, b=0.2, c=-55, d=4)
    
    @classmethod
    def chattering(cls) -> 'IzhikevichNeuron':
        """Create chattering neuron."""
        return cls(a=0.02, b=0.2, c=-50, d=2)
    
    @classmethod
    def fast_spiking(cls) -> 'IzhikevichNeuron':
        """Create fast spiking inhibitory neuron."""
        return cls(a=0.1, b=0.2, c=-65, d=2)
    
    @classmethod
    def low_threshold_spiking(cls) -> 'IzhikevichNeuron':
        """Create low threshold spiking neuron."""
        return cls(a=0.02, b=0.25, c=-65, d=2)
    
    @classmethod
    def thalamo_cortical(cls) -> 'IzhikevichNeuron':
        """Create thalamo-cortical neuron."""
        return cls(a=0.02, b=0.25, c=-65, d=0.05)
    
    def step(self, I_ext: float, dt: float) -> bool:
        """Advance neuron one timestep."""
        # Izhikevich equations (dt typically 1ms for this model)
        self.v += dt * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I_ext)
        self.u += dt * self.a * (self.b * self.v - self.u)
        
        # Spike?
        if self.v >= 30:
            self.v = self.c
            self.u += self.d
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.v = self.c
        self.u = self.b * self.c
    
    @property
    def V(self) -> float:
        """Membrane potential (for compatibility)."""
        return self.v


# =============================================================================
# SYNAPTIC PLASTICITY
# =============================================================================

class STDP:
    """Spike-Timing Dependent Plasticity.
    
    Weight change depends on relative timing of pre and post spikes:
    - Pre before post (causal): LTP (Δw > 0)
    - Post before pre (anti-causal): LTD (Δw < 0)
    
    Δw = A_+ * exp(-Δt/τ_+) if Δt > 0 (pre before post)
    Δw = -A_- * exp(Δt/τ_-) if Δt < 0 (post before pre)
    
    where Δt = t_post - t_pre
    """
    
    def __init__(
        self,
        tau_plus: float = TAU_STDP_PLUS,
        tau_minus: float = TAU_STDP_MINUS,
        A_plus: float = A_PLUS,
        A_minus: float = A_MINUS,
        w_max: float = 1.0,
        w_min: float = 0.0
    ):
        """Initialize STDP rule.
        
        Args:
            tau_plus: LTP time constant [ms]
            tau_minus: LTD time constant [ms]
            A_plus: LTP amplitude
            A_minus: LTD amplitude
            w_max: Maximum weight
            w_min: Minimum weight
        """
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_max = w_max
        self.w_min = w_min
        
    def compute_weight_change(self, delta_t: float) -> float:
        """Compute weight change for given spike timing.
        
        Args:
            delta_t: t_post - t_pre [ms]
        
        Returns:
            Weight change Δw
        """
        if delta_t > 0:
            # Pre before post: LTP
            return self.A_plus * np.exp(-delta_t / self.tau_plus)
        elif delta_t < 0:
            # Post before pre: LTD
            return -self.A_minus * np.exp(delta_t / self.tau_minus)
        else:
            return 0.0
    
    def update_weight(
        self,
        w: float,
        t_pre: float,
        t_post: float
    ) -> float:
        """Update synaptic weight based on spike times.
        
        Args:
            w: Current weight
            t_pre: Presynaptic spike time [ms]
            t_post: Postsynaptic spike time [ms]
        
        Returns:
            Updated weight
        """
        delta_t = t_post - t_pre
        dw = self.compute_weight_change(delta_t)
        
        # Soft bounds
        if dw > 0:
            dw *= (self.w_max - w)
        else:
            dw *= (w - self.w_min)
        
        return np.clip(w + dw, self.w_min, self.w_max)
    
    def learning_window(self, dt_range: Tuple[float, float] = (-100, 100)) -> Tuple[np.ndarray, np.ndarray]:
        """Generate STDP learning window for plotting.
        
        Args:
            dt_range: Range of delta_t values [ms]
        
        Returns:
            Tuple of (delta_t array, weight change array)
        """
        dt = np.linspace(dt_range[0], dt_range[1], 1000)
        dw = np.array([self.compute_weight_change(t) for t in dt])
        return dt, dw


class HebbianPlasticity:
    """Hebbian learning rule.
    
    "Neurons that fire together, wire together."
    
    Rate-based formulation:
    dw/dt = η * pre * post - λ * w
    
    where η is learning rate and λ is weight decay.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
        w_max: float = 1.0,
        w_min: float = 0.0
    ):
        """Initialize Hebbian rule.
        
        Args:
            learning_rate: Learning rate η
            weight_decay: Weight decay λ
            w_max: Maximum weight
            w_min: Minimum weight
        """
        self.eta = learning_rate
        self.lambda_ = weight_decay
        self.w_max = w_max
        self.w_min = w_min
        
    def update_weight(
        self,
        w: float,
        pre_rate: float,
        post_rate: float,
        dt: float = 1.0
    ) -> float:
        """Update weight based on pre and post firing rates.
        
        Args:
            w: Current weight
            pre_rate: Presynaptic firing rate [Hz]
            post_rate: Postsynaptic firing rate [Hz]
            dt: Timestep [ms]
        
        Returns:
            Updated weight
        """
        # Normalize rates
        pre_norm = pre_rate / 100.0  # Assume max ~100 Hz
        post_norm = post_rate / 100.0
        
        dw = (self.eta * pre_norm * post_norm - self.lambda_ * w) * dt
        
        return np.clip(w + dw, self.w_min, self.w_max)


# =============================================================================
# NEURAL NETWORKS
# =============================================================================

class SpikingNetwork:
    """Network of spiking neurons.
    
    Simulates populations of neurons with synaptic connections.
    
    Example:
        >>> net = SpikingNetwork(n_exc=800, n_inh=200)
        >>> net.connect_random(p=0.1)
        >>> spikes = net.run(duration=1000, I_ext=5.0)
    """
    
    def __init__(
        self,
        n_exc: int = 800,
        n_inh: int = 200,
        neuron_model: str = 'izhikevich'
    ):
        """Initialize network.
        
        Args:
            n_exc: Number of excitatory neurons
            n_inh: Number of inhibitory neurons
            neuron_model: 'izhikevich', 'lif', or 'hh'
        """
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        
        # Create neurons
        self.neurons: List[Neuron] = []
        
        if neuron_model == 'izhikevich':
            # Excitatory: regular spiking with some variation
            for i in range(n_exc):
                re = np.random.random()
                a, b = 0.02, 0.2
                c = -65 + 15 * re**2
                d = 8 - 6 * re**2
                self.neurons.append(IzhikevichNeuron(a, b, c, d))
            
            # Inhibitory: fast spiking with variation
            for i in range(n_inh):
                ri = np.random.random()
                a = 0.02 + 0.08 * ri
                b = 0.25 - 0.05 * ri
                c, d = -65, 2
                self.neurons.append(IzhikevichNeuron(a, b, c, d))
                
        elif neuron_model == 'lif':
            for i in range(n_exc):
                self.neurons.append(LeakyIntegrateAndFire())
            for i in range(n_inh):
                self.neurons.append(LeakyIntegrateAndFire(tau_m=5.0))  # Faster
                
        elif neuron_model == 'hh':
            for _ in range(self.n_total):
                self.neurons.append(HodgkinHuxley())
        
        # Synaptic weights (sparse representation would be better for large networks)
        self.weights = np.zeros((self.n_total, self.n_total))
        
        # Synaptic delays [ms]
        self.delays = np.ones((self.n_total, self.n_total))
        
        # STDP
        self.stdp = STDP()
        self.use_stdp = False
        
        # Recording
        self.spike_times: List[Spike] = []
        
    def connect_random(
        self,
        p: float = 0.1,
        w_exc: float = 0.5,
        w_inh: float = -1.0
    ) -> None:
        """Create random sparse connectivity.
        
        Args:
            p: Connection probability
            w_exc: Excitatory weight
            w_inh: Inhibitory weight
        """
        for i in range(self.n_total):
            for j in range(self.n_total):
                if i != j and np.random.random() < p:
                    if i < self.n_exc:
                        self.weights[i, j] = w_exc * np.random.random()
                    else:
                        self.weights[i, j] = w_inh * np.random.random()
                    
                    # Random delays (1-20 ms)
                    self.delays[i, j] = 1 + 19 * np.random.random()
    
    def connect_small_world(
        self,
        k: int = 10,
        p_rewire: float = 0.1,
        w_exc: float = 0.5,
        w_inh: float = -1.0
    ) -> None:
        """Create small-world connectivity (Watts-Strogatz).
        
        Args:
            k: Number of nearest neighbors (even)
            p_rewire: Rewiring probability
            w_exc: Excitatory weight
            w_inh: Inhibitory weight
        """
        # Start with ring lattice
        for i in range(self.n_total):
            for j in range(1, k // 2 + 1):
                # Connect to k nearest neighbors
                target_plus = (i + j) % self.n_total
                target_minus = (i - j) % self.n_total
                
                w = w_exc if i < self.n_exc else w_inh
                self.weights[i, target_plus] = w * np.random.random()
                self.weights[i, target_minus] = w * np.random.random()
        
        # Rewire edges
        for i in range(self.n_total):
            for j in range(self.n_total):
                if self.weights[i, j] != 0 and np.random.random() < p_rewire:
                    # Rewire to random target
                    new_target = np.random.randint(self.n_total)
                    if new_target != i:
                        w = self.weights[i, j]
                        self.weights[i, j] = 0
                        self.weights[i, new_target] = w
    
    def step(self, t: float, I_ext: Union[float, np.ndarray], dt: float) -> List[int]:
        """Advance network one timestep.
        
        Args:
            t: Current time [ms]
            I_ext: External input (scalar or per-neuron array)
            dt: Timestep [ms]
        
        Returns:
            List of neurons that spiked
        """
        if isinstance(I_ext, (int, float)):
            I_ext = np.full(self.n_total, I_ext)
        
        # Compute synaptic input from previous spikes
        I_syn = np.zeros(self.n_total)
        for spike in self.spike_times:
            if t - spike.time > 0:  # Only past spikes
                for post in range(self.n_total):
                    delay = self.delays[spike.neuron_id, post]
                    if abs(t - spike.time - delay) < dt:
                        I_syn[post] += self.weights[spike.neuron_id, post]
        
        # Update neurons
        spiked = []
        for i, neuron in enumerate(self.neurons):
            I_total = I_ext[i] + I_syn[i]
            if neuron.step(I_total, dt):
                spiked.append(i)
                self.spike_times.append(Spike(i, t))
        
        # STDP updates
        if self.use_stdp and spiked:
            self._apply_stdp(t, spiked)
        
        return spiked
    
    def _apply_stdp(self, t: float, spiked: List[int]) -> None:
        """Apply STDP updates for new spikes."""
        for post in spiked:
            # Look for recent presynaptic spikes
            for spike in self.spike_times[-100:]:  # Last 100 spikes
                pre = spike.neuron_id
                if pre != post and self.weights[pre, post] != 0:
                    t_pre = spike.time
                    self.weights[pre, post] = self.stdp.update_weight(
                        self.weights[pre, post], t_pre, t
                    )
    
    def run(
        self,
        duration: float,
        I_ext: Union[float, np.ndarray, Callable[[float], np.ndarray]] = 5.0,
        dt: float = 1.0
    ) -> List[Spike]:
        """Run simulation.
        
        Args:
            duration: Simulation duration [ms]
            I_ext: External input (constant, array, or function of time)
            dt: Timestep [ms]
        
        Returns:
            List of all spikes
        """
        self.spike_times = []
        
        n_steps = int(duration / dt)
        for step in range(n_steps):
            t = step * dt
            
            # Get input
            if callable(I_ext):
                I = I_ext(t)
            else:
                I = I_ext
            
            self.step(t, I, dt)
        
        return self.spike_times
    
    def firing_rates(self, time_window: float = 100.0) -> np.ndarray:
        """Calculate firing rates from spike times.
        
        Args:
            time_window: Time window [ms]
        
        Returns:
            Array of firing rates [Hz]
        """
        rates = np.zeros(self.n_total)
        
        for spike in self.spike_times:
            rates[spike.neuron_id] += 1
        
        # Convert to Hz
        rates *= 1000.0 / time_window
        
        return rates
    
    def raster_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get spike data for raster plot.
        
        Returns:
            Tuple of (times, neuron_ids)
        """
        times = np.array([s.time for s in self.spike_times])
        neurons = np.array([s.neuron_id for s in self.spike_times])
        return times, neurons


# =============================================================================
# BRAIN CONNECTIVITY
# =============================================================================

class BrainGraph:
    """Graph-theoretic analysis of brain connectivity.
    
    Analyzes structural and functional brain networks using
    graph theory metrics.
    """
    
    def __init__(self, adjacency_matrix: np.ndarray, labels: Optional[List[str]] = None):
        """Initialize brain graph.
        
        Args:
            adjacency_matrix: NxN connectivity matrix
            labels: Region labels
        """
        self.A = adjacency_matrix
        self.n_nodes = len(adjacency_matrix)
        self.labels = labels or [f"Region_{i}" for i in range(self.n_nodes)]
        
        # Normalize
        self.W = self.A / (np.max(self.A) + 1e-10)
        
    def degree(self) -> np.ndarray:
        """Calculate node degree (number of connections).
        
        Returns:
            Degree of each node
        """
        return np.sum(self.A > 0, axis=1) + np.sum(self.A > 0, axis=0)
    
    def strength(self) -> np.ndarray:
        """Calculate node strength (sum of weights).
        
        Returns:
            Strength of each node
        """
        return np.sum(self.W, axis=1) + np.sum(self.W, axis=0)
    
    def clustering_coefficient(self) -> np.ndarray:
        """Calculate local clustering coefficient.
        
        Returns:
            Clustering coefficient for each node
        """
        cc = np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            neighbors = np.where((self.A[i, :] > 0) | (self.A[:, i] > 0))[0]
            k = len(neighbors)
            
            if k < 2:
                cc[i] = 0
                continue
            
            # Count triangles
            triangles = 0
            for j in neighbors:
                for l in neighbors:
                    if j < l and (self.A[j, l] > 0 or self.A[l, j] > 0):
                        triangles += 1
            
            cc[i] = 2 * triangles / (k * (k - 1))
        
        return cc
    
    def global_clustering(self) -> float:
        """Calculate global clustering coefficient.
        
        Returns:
            Average clustering coefficient
        """
        return np.mean(self.clustering_coefficient())
    
    def shortest_path_length(self) -> np.ndarray:
        """Calculate shortest path lengths (Floyd-Warshall).
        
        Returns:
            NxN matrix of shortest paths
        """
        # Convert weights to distances (inverse)
        # Avoid division by zero by using a safe inverse
        with np.errstate(divide='ignore', invalid='ignore'):
            D = np.where(self.W > 0, 1.0 / self.W, np.inf)
        np.fill_diagonal(D, 0)
        
        # Floyd-Warshall
        for k in range(self.n_nodes):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if D[i, k] + D[k, j] < D[i, j]:
                        D[i, j] = D[i, k] + D[k, j]
        
        return D
    
    def characteristic_path_length(self) -> float:
        """Calculate characteristic path length.
        
        Returns:
            Average shortest path length
        """
        D = self.shortest_path_length()
        # Exclude infinite and self paths
        mask = (D < np.inf) & (D > 0)
        if np.sum(mask) == 0:
            return np.inf
        return np.mean(D[mask])
    
    def betweenness_centrality(self) -> np.ndarray:
        """Calculate betweenness centrality.
        
        Returns:
            Betweenness centrality for each node
        """
        bc = np.zeros(self.n_nodes)
        D = self.shortest_path_length()
        
        for s in range(self.n_nodes):
            for t in range(self.n_nodes):
                if s != t and D[s, t] < np.inf:
                    for v in range(self.n_nodes):
                        if v != s and v != t:
                            if D[s, v] + D[v, t] == D[s, t]:
                                bc[v] += 1
        
        # Normalize
        norm = (self.n_nodes - 1) * (self.n_nodes - 2)
        if norm > 0:
            bc /= norm
        
        return bc
    
    def modularity(self, communities: List[int]) -> float:
        """Calculate modularity for given community assignment.
        
        Args:
            communities: Community label for each node
        
        Returns:
            Modularity Q
        """
        m = np.sum(self.W) / 2
        if m == 0:
            return 0
        
        k = self.strength()
        Q = 0
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if communities[i] == communities[j]:
                    Q += self.W[i, j] - k[i] * k[j] / (2 * m)
        
        return Q / (2 * m)
    
    def small_worldness(self, n_random: int = 10) -> float:
        """Calculate small-worldness coefficient.
        
        σ = (C/C_rand) / (L/L_rand)
        
        where C is clustering and L is path length.
        
        Args:
            n_random: Number of random graphs for comparison
        
        Returns:
            Small-worldness coefficient (σ > 1 indicates small-world)
        """
        C = self.global_clustering()
        L = self.characteristic_path_length()
        
        # Generate random graphs with same degree sequence
        C_rand = 0
        L_rand = 0
        
        for _ in range(n_random):
            # Simple random rewiring
            A_rand = self.A.copy()
            n_edges = int(np.sum(A_rand > 0))
            
            for _ in range(n_edges):
                # Find two edges and swap
                edges = np.argwhere(A_rand > 0)
                if len(edges) < 2:
                    break
                idx = np.random.choice(len(edges), 2, replace=False)
                i1, j1 = edges[idx[0]]
                i2, j2 = edges[idx[1]]
                
                # Swap
                A_rand[i1, j1], A_rand[i2, j2] = 0, 0
                A_rand[i1, j2], A_rand[i2, j1] = 1, 1
            
            rand_graph = BrainGraph(A_rand)
            C_rand += rand_graph.global_clustering()
            L_rand += rand_graph.characteristic_path_length()
        
        C_rand /= n_random
        L_rand /= n_random
        
        if C_rand == 0 or L_rand == 0:
            return 1.0
        
        return (C / C_rand) / (L / L_rand)
    
    def hub_nodes(self, threshold: float = 0.9) -> List[int]:
        """Identify hub nodes based on degree centrality.
        
        Args:
            threshold: Percentile threshold for hub classification
        
        Returns:
            List of hub node indices
        """
        degree = self.degree()
        cutoff = np.percentile(degree, threshold * 100)
        return list(np.where(degree >= cutoff)[0])


# =============================================================================
# NEURAL CODING
# =============================================================================

def spike_count_rate(spikes: List[Spike], neuron_id: int, window: Tuple[float, float]) -> float:
    """Calculate firing rate by spike counting.
    
    Args:
        spikes: List of spikes
        neuron_id: Neuron to analyze
        window: Time window (start, end) [ms]
    
    Returns:
        Firing rate [Hz]
    """
    count = sum(1 for s in spikes if s.neuron_id == neuron_id and window[0] <= s.time < window[1])
    duration = (window[1] - window[0]) / 1000  # Convert to seconds
    return count / duration if duration > 0 else 0


def interspike_intervals(spikes: List[Spike], neuron_id: int) -> np.ndarray:
    """Calculate interspike intervals.
    
    Args:
        spikes: List of spikes
        neuron_id: Neuron to analyze
    
    Returns:
        Array of ISIs [ms]
    """
    times = sorted([s.time for s in spikes if s.neuron_id == neuron_id])
    if len(times) < 2:
        return np.array([])
    return np.diff(times)


def coefficient_of_variation(isis: np.ndarray) -> float:
    """Calculate coefficient of variation of ISIs.
    
    CV = std(ISI) / mean(ISI)
    
    CV = 1 for Poisson process
    CV < 1 for regular spiking
    CV > 1 for bursting
    
    Args:
        isis: Interspike intervals
    
    Returns:
        Coefficient of variation
    """
    if len(isis) < 2:
        return 0.0
    return np.std(isis) / np.mean(isis)


def fano_factor(spike_counts: np.ndarray) -> float:
    """Calculate Fano factor.
    
    FF = var(count) / mean(count)
    
    FF = 1 for Poisson process
    
    Args:
        spike_counts: Spike counts in multiple windows
    
    Returns:
        Fano factor
    """
    if len(spike_counts) < 2 or np.mean(spike_counts) == 0:
        return 0.0
    return np.var(spike_counts) / np.mean(spike_counts)


def spike_train_correlation(
    spikes1: List[float],
    spikes2: List[float],
    bin_size: float = 10.0,
    max_lag: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate cross-correlation between spike trains.
    
    Args:
        spikes1: First spike train times [ms]
        spikes2: Second spike train times [ms]
        bin_size: Bin size [ms]
        max_lag: Maximum lag [ms]
    
    Returns:
        Tuple of (lags, correlation)
    """
    if not spikes1 or not spikes2:
        return np.array([]), np.array([])
    
    lags = np.arange(-max_lag, max_lag + bin_size, bin_size)
    corr = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        for t1 in spikes1:
            for t2 in spikes2:
                dt = t2 - t1 - lag
                if abs(dt) < bin_size / 2:
                    corr[i] += 1
    
    # Normalize
    norm = np.sqrt(len(spikes1) * len(spikes2))
    if norm > 0:
        corr /= norm
    
    return lags, corr


def information_rate(
    stimulus: np.ndarray,
    response: np.ndarray,
    n_bins: int = 10
) -> float:
    """Estimate mutual information between stimulus and response.
    
    I(S; R) = H(R) - H(R|S)
    
    Args:
        stimulus: Stimulus values
        response: Response values (e.g., spike counts)
        n_bins: Number of bins for discretization
    
    Returns:
        Mutual information [bits]
    """
    # Discretize
    s_bins = np.linspace(np.min(stimulus), np.max(stimulus), n_bins + 1)
    r_bins = np.linspace(np.min(response), np.max(response), n_bins + 1)
    
    s_discrete = np.digitize(stimulus, s_bins[:-1]) - 1
    r_discrete = np.digitize(response, r_bins[:-1]) - 1
    
    # Joint and marginal distributions
    joint = np.zeros((n_bins, n_bins))
    for s, r in zip(s_discrete, r_discrete):
        s = min(s, n_bins - 1)
        r = min(r, n_bins - 1)
        joint[s, r] += 1
    joint /= len(stimulus)
    
    p_s = np.sum(joint, axis=1)
    p_r = np.sum(joint, axis=0)
    
    # Mutual information
    MI = 0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and p_s[i] > 0 and p_r[j] > 0:
                MI += joint[i, j] * np.log2(joint[i, j] / (p_s[i] * p_r[j]))
    
    return MI


# =============================================================================
# COGNITIVE MODELS
# =============================================================================

class DriftDiffusionModel:
    """Drift-Diffusion Model for decision making.
    
    Models evidence accumulation to threshold:
    dx = v*dt + s*dW
    
    Decision when x reaches +a (choice A) or -a (choice B).
    
    Parameters:
    - v: drift rate (evidence strength)
    - a: boundary height (speed-accuracy tradeoff)
    - s: noise (diffusion coefficient)
    - t0: non-decision time
    """
    
    def __init__(
        self,
        v: float = 0.3,
        a: float = 1.0,
        s: float = 0.1,
        t0: float = 0.3,
        z: float = 0.5  # Starting point (fraction of a)
    ):
        """Initialize DDM.
        
        Args:
            v: Drift rate
            a: Boundary separation
            s: Noise standard deviation
            t0: Non-decision time [s]
            z: Starting point (fraction of boundary)
        """
        self.v = v
        self.a = a
        self.s = s
        self.t0 = t0
        self.z = z
        
    def simulate_trial(self, dt: float = 0.001, max_time: float = 10.0) -> Tuple[int, float]:
        """Simulate single trial.
        
        Args:
            dt: Timestep [s]
            max_time: Maximum decision time [s]
        
        Returns:
            Tuple of (choice [1 or -1], reaction time [s])
        """
        x = self.z * self.a  # Start between boundaries
        t = 0
        
        while t < max_time:
            # Drift + noise
            dx = self.v * dt + self.s * np.sqrt(dt) * np.random.randn()
            x += dx
            t += dt
            
            # Check boundaries
            if x >= self.a:
                return 1, t + self.t0
            elif x <= -self.a:
                return -1, t + self.t0
        
        # Timeout - choose based on position
        return (1 if x > 0 else -1), max_time + self.t0
    
    def simulate_experiment(
        self,
        n_trials: int = 1000,
        dt: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate multiple trials.
        
        Args:
            n_trials: Number of trials
            dt: Timestep [s]
        
        Returns:
            Tuple of (choices array, reaction times array)
        """
        choices = np.zeros(n_trials)
        rts = np.zeros(n_trials)
        
        for i in range(n_trials):
            choices[i], rts[i] = self.simulate_trial(dt)
        
        return choices, rts
    
    def accuracy(self, n_trials: int = 1000) -> float:
        """Calculate accuracy (proportion choosing boundary +a).
        
        Args:
            n_trials: Number of simulation trials
        
        Returns:
            Accuracy [0-1]
        """
        choices, _ = self.simulate_experiment(n_trials)
        return np.mean(choices == 1)
    
    def mean_rt(self, n_trials: int = 1000) -> float:
        """Calculate mean reaction time.
        
        Args:
            n_trials: Number of simulation trials
        
        Returns:
            Mean RT [s]
        """
        _, rts = self.simulate_experiment(n_trials)
        return np.mean(rts)


class WorkingMemory:
    """Working memory model with capacity limit.
    
    Based on slot-based models with interference.
    """
    
    def __init__(self, capacity: int = 4, decay_rate: float = 0.1):
        """Initialize working memory.
        
        Args:
            capacity: Maximum number of items (typically 4±1)
            decay_rate: Decay rate per second
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        
        # Memory slots: (item, strength, encoding_time)
        self.slots: List[Tuple[any, float, float]] = []
        self.time = 0.0
        
    def encode(self, item: any, strength: float = 1.0) -> bool:
        """Encode item into working memory.
        
        Args:
            item: Item to encode
            strength: Initial encoding strength
        
        Returns:
            True if successfully encoded
        """
        # Check if already in memory
        for i, (stored, s, t) in enumerate(self.slots):
            if stored == item:
                # Refresh
                self.slots[i] = (item, strength, self.time)
                return True
        
        # Add new item
        if len(self.slots) < self.capacity:
            self.slots.append((item, strength, self.time))
            return True
        else:
            # Replace weakest item
            strengths = [self._current_strength(s, t) for _, s, t in self.slots]
            weakest = np.argmin(strengths)
            self.slots[weakest] = (item, strength, self.time)
            return True
    
    def _current_strength(self, initial_strength: float, encoding_time: float) -> float:
        """Calculate current strength with decay."""
        elapsed = self.time - encoding_time
        return initial_strength * np.exp(-self.decay_rate * elapsed)
    
    def retrieve(self, item: any) -> Optional[float]:
        """Attempt to retrieve item from memory.
        
        Args:
            item: Item to retrieve
        
        Returns:
            Retrieval strength if found, None otherwise
        """
        for stored, strength, enc_time in self.slots:
            if stored == item:
                return self._current_strength(strength, enc_time)
        return None
    
    def recall_probability(self, item: any, threshold: float = 0.3) -> float:
        """Calculate probability of successful recall.
        
        Args:
            item: Item to recall
            threshold: Strength threshold for recall
        
        Returns:
            Recall probability [0-1]
        """
        strength = self.retrieve(item)
        if strength is None:
            return 0.0
        
        # Sigmoid function around threshold
        return 1 / (1 + np.exp(-10 * (strength - threshold)))
    
    def update_time(self, dt: float) -> None:
        """Advance time.
        
        Args:
            dt: Time increment [s]
        """
        self.time += dt
        
        # Remove items below threshold
        self.slots = [
            (item, s, t) for item, s, t in self.slots
            if self._current_strength(s, t) > 0.01
        ]
    
    def contents(self) -> List[Tuple[any, float]]:
        """Get current memory contents.
        
        Returns:
            List of (item, current_strength) tuples
        """
        return [
            (item, self._current_strength(s, t))
            for item, s, t in self.slots
        ]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'C_M', 'R_M', 'TAU_M', 'E_NA', 'E_K', 'E_LEAK',
    'G_NA', 'G_K', 'G_LEAK', 'V_REST', 'V_THRESHOLD',
    
    # Enums
    'NeuronType', 'SynapseType', 'PlasticityRule', 'BrainRegion',
    
    # Data classes
    'Spike', 'SynapticConnection', 'NeuronState',
    
    # Neuron models
    'Neuron', 'LeakyIntegrateAndFire', 'HodgkinHuxley', 'IzhikevichNeuron',
    
    # Plasticity
    'STDP', 'HebbianPlasticity',
    
    # Networks
    'SpikingNetwork', 'BrainGraph',
    
    # Neural coding functions
    'spike_count_rate', 'interspike_intervals', 'coefficient_of_variation',
    'fano_factor', 'spike_train_correlation', 'information_rate',
    
    # Cognitive models
    'DriftDiffusionModel', 'WorkingMemory',
]

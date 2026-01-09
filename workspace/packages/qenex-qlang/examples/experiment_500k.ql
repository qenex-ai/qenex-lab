# experiment_500k.ql
# Assessing Habitability at 500 Kelvin
# ==========================================

print "=================================================="
print "       QENEX: EXTREME ENVIRONMENT PROTOCOL        "
print "       Target Temperature: 500 Kelvin             "
print "=================================================="

# Mapping Real Temperature to Simulation Units
# Assumption: T_c (Ising) approx 2.27 corresponds to a critical phase transition.
# If we map 300K (habitable) to T=2.0:
# 500K / 300K * 2.0 = 3.33
define temp_sim = 3.33

# --------------------------------------------------------
# STAGE 1: PHYSICS (Thermodynamic Phase)
# --------------------------------------------------------
print "\n[STAGE 1] PHYSICS: Checking Phase Stability at T="
print $temp_sim

define lattice_size = 32
define sweeps = 1000

simulate physics $lattice_size $sweeps $temp_sim

# Check Order Parameter (Magnetization)
# At T > 2.27, M should drop to ~0 (Disordered/Gas/Plasma)
print "   Magnetization (Order Parameter): "
print $last_magnetization

if abs($last_magnetization) < 0.2:
    print "❌ PHYSICS FAILURE: System is Disordered (High Entropy)."
    print "   Environment is likely gaseous or plasma. Structural integrity compromised."
    define physics_ok = 0
else:
    print "✅ PHYSICS PASS: System maintains ordered phase."
    define physics_ok = 1
end

# --------------------------------------------------------
# STAGE 2: CHEMISTRY (Molecular Stability)
# --------------------------------------------------------
# Even if physics fails, let's try to optimize a molecule to see if bonds hold.
# (In this simulation, Geometry Opt is T-independent electronic structure,
# but we interpret the result in context).

if $physics_ok:
    print "\n[STAGE 2] CHEMISTRY: Optimizing Water..."
    optimize geometry O 0,0,0 H 1.8,0,0 H 0.5,1.5,0 sto-3g
    if $last_energy < -74.0:
        print "✅ CHEMISTRY PASS: Molecules stable."
        define chem_ok = 1
    else:
        print "❌ CHEMISTRY FAILURE: Unstable."
        define chem_ok = 0
    end
else:
    print "\n[STAGE 2] CHEMISTRY: Skipped due to phase instability."
    define chem_ok = 0
end

# --------------------------------------------------------
# STAGE 3: BIOLOGY (Protein Folding)
# --------------------------------------------------------
# Protein folding is highly sensitive to Temperature.
# We pass the simulation temperature directly.

if $chem_ok:
    print "\n[STAGE 3] BIOLOGY: Attempting Protein Folding..."
    define sequence = "HHPPHHPH"
    
    # Passing the high temperature to the Monte Carlo folder
    simulate biology folding $sequence temperature=$temp_sim
    
    print "   Final Folding Energy: "
    print $last_energy
    
    # Native state is usually around -3.0 or lower
    if $last_energy < -2.0:
        print "✅ BIOLOGY PASS: Protein folded to native state."
        define bio_ok = 1
    else:
        print "❌ BIOLOGY FAILURE: Protein denatured (Thermal Fluctuation)."
        define bio_ok = 0
    end
else:
    print "\n[STAGE 3] BIOLOGY: Life impossible due to precursor failure."
    define bio_ok = 0
end

print "\n=================================================="
print "FINAL VERDICT FOR 500 KELVIN:"
if $bio_ok:
    print "HABITABLE."
else:
    print "UNINHABITABLE."
end
print "=================================================="

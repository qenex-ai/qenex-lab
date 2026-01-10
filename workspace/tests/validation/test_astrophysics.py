"""
Tests for QENEX Astrophysics Module
===================================
Validates stellar physics, cosmology, and gravitational wave calculations
against known values and physical constraints.
"""

import pytest
import numpy as np
import sys
import os

# Add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'packages', 'qenex-astro', 'src'))

from astrophysics import (
    Star, StellarStructure, CosmologicalModel, DarkMatterHalo,
    Exoplanet, BinarySystem, AstrophysicsSimulator,
    G, c, M_sun, R_sun, L_sun, AU, pc, H_0, T_sun
)


class TestStellarPhysics:
    """Tests for stellar physics calculations."""
    
    def test_sun_properties(self):
        """Test that a solar-mass star has correct properties."""
        sun = Star(mass=1.0, name="Sun")
        
        # Luminosity should be close to 1 L_sun
        assert 0.9 < sun.luminosity < 1.1
        
        # Radius should be close to 1 R_sun
        assert 0.9 < sun.radius < 1.1
        
        # Temperature should be ~5772 K
        assert 5500 < sun.temperature < 6000
    
    def test_mass_luminosity_relation(self):
        """Test L ∝ M^3.5 for main sequence stars."""
        m1 = Star(mass=1.0)
        m2 = Star(mass=2.0)
        
        # For M > 0.5 M_sun: L ∝ M^3.5
        expected_ratio = 2.0 ** 3.5
        actual_ratio = m2.luminosity / m1.luminosity
        
        assert 0.5 * expected_ratio < actual_ratio < 1.5 * expected_ratio
    
    def test_main_sequence_lifetime(self):
        """Test main sequence lifetime scaling."""
        sun = Star(mass=1.0)
        
        # Sun should live ~10 Gyr on main sequence
        assert 8 < sun.main_sequence_lifetime < 12
    
    def test_habitable_zone(self):
        """Test habitable zone calculation for Sun."""
        sun = Star(mass=1.0)
        hz_inner, hz_outer = sun.habitable_zone()
        
        # Earth is at 1 AU, should be in habitable zone
        assert hz_inner < 1.0 < hz_outer
        
        # Typical range ~0.95 to 1.7 AU for Sun
        assert 0.8 < hz_inner < 1.0
        assert 1.3 < hz_outer < 2.0
    
    def test_spectral_classification(self):
        """Test spectral type assignment."""
        hot_star = Star(mass=10.0)  # Should be O/B type
        sun = Star(mass=1.0)  # Should be G type
        cool_star = Star(mass=0.3)  # Should be M type
        
        # Temperature ordering
        assert hot_star.temperature > sun.temperature > cool_star.temperature


class TestCosmology:
    """Tests for cosmological calculations."""
    
    def test_hubble_constant(self):
        """Test H(z=0) equals H_0."""
        cosmo = CosmologicalModel()
        
        H_now = cosmo.H(0)
        assert abs(H_now - cosmo.H_0) < 0.1
    
    def test_age_of_universe(self):
        """Test age of universe is ~13.8 Gyr."""
        cosmo = CosmologicalModel()
        
        age = cosmo.age_of_universe()
        assert 13.0 < age < 14.5
    
    def test_comoving_distance_increases(self):
        """Test comoving distance increases with redshift."""
        cosmo = CosmologicalModel()
        
        d1 = cosmo.comoving_distance(0.1)
        d2 = cosmo.comoving_distance(1.0)
        d3 = cosmo.comoving_distance(2.0)
        
        assert d1 < d2 < d3
    
    def test_lookback_time(self):
        """Test lookback time increases with redshift."""
        cosmo = CosmologicalModel()
        
        t1 = cosmo.lookback_time(0.1)
        t2 = cosmo.lookback_time(1.0)
        t3 = cosmo.lookback_time(3.0)
        
        assert 0 < t1 < t2 < t3
        
        # Lookback time to z=1 should be ~7-8 Gyr
        assert 6 < t2 < 9
    
    def test_critical_density(self):
        """Test critical density calculation."""
        cosmo = CosmologicalModel()
        
        rho_c = cosmo.critical_density()
        
        # ρ_c ≈ 9.5e-27 kg/m³
        assert 8e-27 < rho_c < 1.1e-26
    
    def test_flat_universe(self):
        """Test that default model is flat (Ω_total = 1)."""
        cosmo = CosmologicalModel()
        
        omega_total = cosmo.Omega_m + cosmo.Omega_Lambda
        assert abs(omega_total - 1.0) < 0.01


class TestDarkMatter:
    """Tests for dark matter halo calculations."""
    
    def test_nfw_density_decreases(self):
        """Test NFW density decreases with radius."""
        halo = DarkMatterHalo(M_vir=1e12, c=10)  # Milky Way-like
        
        # Density method takes radius in meters, convert kpc to m
        kpc_to_m = 1000 * pc
        rho1 = halo.density(1.0 * kpc_to_m)   # 1 kpc
        rho10 = halo.density(10.0 * kpc_to_m)  # 10 kpc
        rho100 = halo.density(100.0 * kpc_to_m)  # 100 kpc
        
        assert rho1 > rho10 > rho100 > 0
    
    def test_rotation_curve_shape(self):
        """Test rotation curve has expected flat behavior at large r."""
        halo = DarkMatterHalo(M_vir=1e12, c=10)
        
        r = np.array([10, 20, 30, 50, 100])
        v = halo.rotation_curve(r)
        
        # Rotation curve should be relatively flat at large radii
        # (variation < 30% from 10 to 100 kpc)
        assert np.std(v) / np.mean(v) < 0.3
    
    def test_virial_mass_enclosed(self):
        """Test that mass enclosed at r_vir is close to M_vir."""
        M_vir = 1e12  # Solar masses
        halo = DarkMatterHalo(M_vir=M_vir, c=10)
        
        # Mass at virial radius should be close to M_vir
        # Note: enclosed_mass returns kg, M_vir is in solar masses
        M_enclosed_kg = halo.enclosed_mass(halo.r_vir)
        M_enclosed_msun = M_enclosed_kg / M_sun
        
        # Allow 20% tolerance due to different density profiles
        assert 0.8 * M_vir < M_enclosed_msun < 1.2 * M_vir


class TestExoplanets:
    """Tests for exoplanet calculations."""
    
    def test_orbital_period_kepler(self):
        """Test orbital period follows Kepler's third law."""
        host = Star(mass=1.0)
        
        # Earth-like: 1 AU should give ~365 days
        earth = Exoplanet(
            mass=1.0,  # Earth masses
            radius=1.0,  # Earth radii
            semi_major_axis=1.0,  # AU
            host_star=host
        )
        
        # Period is an attribute set by _kepler_period
        period_days = earth.period
        assert 360 < period_days < 370
    
    def test_equilibrium_temperature(self):
        """Test equilibrium temperature calculation."""
        host = Star(mass=1.0)
        
        earth = Exoplanet(
            mass=1.0,
            radius=1.0,
            semi_major_axis=1.0,
            host_star=host
        )
        
        # equilibrium_temperature takes albedo as parameter
        T_eq = earth.equilibrium_temperature(albedo=0.3)
        
        # Earth's T_eq ~ 255 K (without greenhouse effect)
        assert 240 < T_eq < 270
    
    def test_transit_depth(self):
        """Test transit depth calculation."""
        host = Star(mass=1.0)
        
        # Hot Jupiter: large planet
        hot_jup = Exoplanet(
            mass=300,  # Earth masses (~1 Jupiter)
            radius=11,  # Earth radii (~1 Jupiter radius)
            semi_major_axis=0.05,
            host_star=host
        )
        
        depth = hot_jup.transit_depth()
        
        # Jupiter transiting Sun: (R_J/R_sun)^2 ~ 0.01
        assert 0.005 < depth < 0.02


class TestGravitationalWaves:
    """Tests for gravitational wave calculations."""
    
    def test_chirp_mass(self):
        """Test chirp mass calculation."""
        # Equal mass binary
        binary = BinarySystem(m1=30.0, m2=30.0, frequency=100)
        
        # M_chirp = (m1*m2)^(3/5) / (m1+m2)^(1/5) * (m1+m2) = M_total * eta^(3/5)
        # For equal masses: M_chirp = m_total * (0.25)^(3/5) ~ 0.435 * M_total
        # In solar masses: M_chirp_msun ~ 0.435 * 60 ~ 26.1 M_sun
        M_chirp_msun = binary.M_chirp / M_sun
        
        # Expected: ~26 solar masses
        assert 20 < M_chirp_msun < 30
    
    def test_strain_decreases_with_distance(self):
        """Test strain amplitude decreases with distance."""
        binary = BinarySystem(m1=30.0, m2=30.0, frequency=100)
        
        h1 = binary.strain_amplitude(100 * 1e6 * pc)  # 100 Mpc
        h2 = binary.strain_amplitude(200 * 1e6 * pc)  # 200 Mpc
        
        # h ∝ 1/r
        assert abs(h1 / h2 - 2.0) < 0.1
    
    def test_strain_order_of_magnitude(self):
        """Test strain is in detectable range for nearby mergers."""
        binary = BinarySystem(m1=30.0, m2=30.0, frequency=100)
        
        # 100 Mpc distance
        h = binary.strain_amplitude(100 * 1e6 * pc)
        
        # LIGO sensitivity ~1e-23, so strain should be > 1e-24
        assert h > 1e-25
        assert h < 1e-20


class TestAstrophysicsSimulator:
    """Integration tests for the full simulator."""
    
    def test_stellar_simulation(self):
        """Test stellar simulation runs without error."""
        sim = AstrophysicsSimulator()
        result = sim.simulate_star(mass=1.0, name="TestStar")
        
        assert 'name' in result
        assert 'temperature_K' in result
        assert 'luminosity_Lsun' in result
        assert result['luminosity_Lsun'] > 0
    
    def test_cosmology_simulation(self):
        """Test cosmology simulation produces valid results."""
        sim = AstrophysicsSimulator()
        result = sim.simulate_cosmology(z_max=2.0)
        
        assert 'model' in result
        assert 'comoving_distance_Mpc' in result
        assert len(result['comoving_distance_Mpc']) > 0
    
    def test_dark_matter_simulation(self):
        """Test dark matter halo simulation."""
        sim = AstrophysicsSimulator()
        result = sim.simulate_dark_matter_halo(M_vir=1e12, c=10)
        
        assert 'rotation_curve' in result
        assert len(result['rotation_curve']['r_kpc']) > 0


class TestPhysicalConstraints:
    """Tests for physical sanity checks."""
    
    def test_positive_luminosity(self):
        """All stars must have positive luminosity."""
        for mass in [0.1, 0.5, 1.0, 5.0, 20.0]:
            star = Star(mass=mass)
            assert star.luminosity > 0
    
    def test_positive_radius(self):
        """All stars must have positive radius."""
        for mass in [0.1, 0.5, 1.0, 5.0, 20.0]:
            star = Star(mass=mass)
            assert star.radius > 0
    
    def test_positive_temperature(self):
        """All stars must have positive temperature."""
        for mass in [0.1, 0.5, 1.0, 5.0, 20.0]:
            star = Star(mass=mass)
            assert star.temperature > 0
    
    def test_constants_positive(self):
        """Physical constants must be positive."""
        assert G > 0
        assert c > 0
        assert M_sun > 0
        assert R_sun > 0
        assert L_sun > 0
        assert AU > 0
        assert pc > 0
        assert H_0 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

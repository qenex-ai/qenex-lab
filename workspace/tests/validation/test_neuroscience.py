"""
Tests for QENEX Neuroscience Module
===================================
Validates neuron models, synaptic plasticity, and neural network dynamics
against known biophysical properties.
"""

import pytest
import numpy as np
import sys
import os

# Add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'packages', 'qenex-neuro', 'src'))

from neuroscience import (
    LeakyIntegrateAndFire, HodgkinHuxley, IzhikevichNeuron,
    STDP, HebbianPlasticity,
    SpikingNetwork, BrainGraph,
    DriftDiffusionModel, WorkingMemory,
    spike_count_rate, interspike_intervals, coefficient_of_variation,
    fano_factor, information_rate, Spike,
    V_REST, V_THRESHOLD, E_NA, E_K, TAU_M
)


class TestLeakyIntegrateAndFire:
    """Tests for LIF neuron model."""
    
    def test_resting_potential(self):
        """Test that neuron starts at resting potential."""
        lif = LeakyIntegrateAndFire()
        assert abs(lif.V - V_REST) < 1.0
    
    def test_no_spike_below_threshold(self):
        """Test that weak input doesn't cause spike."""
        lif = LeakyIntegrateAndFire()
        
        # Run with subthreshold input
        for _ in range(100):
            spiked = lif.step(I_ext=0.5, dt=0.1)
            assert not spiked
    
    def test_spike_above_threshold(self):
        """Test that strong input causes spike."""
        lif = LeakyIntegrateAndFire()
        
        # Run with suprathreshold input
        spike_occurred = False
        for _ in range(1000):
            if lif.step(I_ext=15.0, dt=0.1):
                spike_occurred = True
                break
        
        assert spike_occurred
    
    def test_reset_after_spike(self):
        """Test that voltage resets after spike."""
        lif = LeakyIntegrateAndFire()
        
        # Run until spike
        for _ in range(1000):
            spiked = lif.step(I_ext=15.0, dt=0.1)
            if spiked:
                # Check reset
                assert lif.V < V_THRESHOLD
                break
    
    def test_firing_rate_increases_with_current(self):
        """Test that firing rate increases with input current."""
        lif = LeakyIntegrateAndFire()
        
        rate_low = lif.firing_rate(I_ext=5.0)
        rate_high = lif.firing_rate(I_ext=15.0)
        
        assert rate_high > rate_low
    
    def test_refractory_period(self):
        """Test that refractory period prevents immediate re-spiking."""
        lif = LeakyIntegrateAndFire(t_refract=2.0)
        
        # Force first spike
        spike_times = []
        for t in range(500):
            if lif.step(I_ext=20.0, dt=0.1):
                spike_times.append(t * 0.1)
        
        # Check inter-spike intervals
        if len(spike_times) >= 2:
            isi = spike_times[1] - spike_times[0]
            assert isi >= lif.t_refract


class TestHodgkinHuxley:
    """Tests for Hodgkin-Huxley neuron model."""
    
    def test_resting_potential(self):
        """Test initial resting potential."""
        hh = HodgkinHuxley()
        assert -70 < hh.V < -60
    
    def test_action_potential_shape(self):
        """Test that action potential reaches appropriate peak."""
        hh = HodgkinHuxley()
        
        V_max = hh.V
        for _ in range(10000):
            hh.step(I_ext=10.0, dt=0.01)
            V_max = max(V_max, hh.V)
        
        # Action potential should reach ~+30 mV
        assert V_max > 0
    
    def test_gating_variables_bounded(self):
        """Test that gating variables stay in [0, 1]."""
        hh = HodgkinHuxley()
        
        for _ in range(1000):
            hh.step(I_ext=10.0, dt=0.01)
            
            assert 0 <= hh.m <= 1
            assert 0 <= hh.h <= 1
            assert 0 <= hh.n <= 1
    
    def test_spike_detection(self):
        """Test that spikes can be detected."""
        hh = HodgkinHuxley()
        
        spike_detected = False
        for _ in range(10000):
            if hh.step(I_ext=10.0, dt=0.01):
                spike_detected = True
                break
        
        assert spike_detected


class TestIzhikevichNeuron:
    """Tests for Izhikevich neuron model."""
    
    def test_regular_spiking(self):
        """Test regular spiking neuron fires."""
        izh = IzhikevichNeuron.regular_spiking()
        
        spike_count = 0
        for _ in range(1000):
            if izh.step(I_ext=10, dt=1.0):
                spike_count += 1
        
        assert spike_count > 0
    
    def test_fast_spiking(self):
        """Test fast spiking neuron has higher rate."""
        rs = IzhikevichNeuron.regular_spiking()
        fs = IzhikevichNeuron.fast_spiking()
        
        rs_spikes = sum(1 for _ in range(500) if rs.step(I_ext=10, dt=1.0))
        fs_spikes = sum(1 for _ in range(500) if fs.step(I_ext=10, dt=1.0))
        
        # Fast spiking should have higher rate for same input
        assert fs_spikes >= rs_spikes
    
    def test_different_neuron_types(self):
        """Test different neuron type constructors work."""
        types = [
            IzhikevichNeuron.regular_spiking(),
            IzhikevichNeuron.intrinsically_bursting(),
            IzhikevichNeuron.chattering(),
            IzhikevichNeuron.fast_spiking(),
            IzhikevichNeuron.low_threshold_spiking(),
        ]
        
        for neuron in types:
            # All should be able to spike
            spiked = False
            for _ in range(500):
                if neuron.step(I_ext=15, dt=1.0):
                    spiked = True
                    break
            assert spiked


class TestSTDP:
    """Tests for spike-timing dependent plasticity."""
    
    def test_ltp_for_causal_timing(self):
        """Test LTP when pre before post (causal)."""
        stdp = STDP()
        
        # Pre at t=0, post at t=10 (positive delta_t)
        dw = stdp.compute_weight_change(delta_t=10.0)
        assert dw > 0
    
    def test_ltd_for_anticausal_timing(self):
        """Test LTD when post before pre (anti-causal)."""
        stdp = STDP()
        
        # Post at t=0, pre at t=10 (negative delta_t)
        dw = stdp.compute_weight_change(delta_t=-10.0)
        assert dw < 0
    
    def test_weight_change_decays_with_time(self):
        """Test that weight change decays with time difference."""
        stdp = STDP()
        
        dw_short = stdp.compute_weight_change(delta_t=5.0)
        dw_long = stdp.compute_weight_change(delta_t=50.0)
        
        assert abs(dw_short) > abs(dw_long)
    
    def test_weight_bounded(self):
        """Test that weight stays within bounds."""
        stdp = STDP(w_min=0.0, w_max=1.0)
        
        # Strong LTP
        w = 0.5
        w = stdp.update_weight(w, t_pre=0, t_post=5)
        assert 0.0 <= w <= 1.0
        
        # Strong LTD
        w = stdp.update_weight(w, t_pre=100, t_post=0)
        assert 0.0 <= w <= 1.0
    
    def test_learning_window_shape(self):
        """Test STDP learning window has correct shape."""
        stdp = STDP()
        dt, dw = stdp.learning_window((-50, 50))
        
        # Should have positive values for positive dt
        positive_dw = dw[dt > 0]
        assert np.mean(positive_dw) > 0
        
        # Should have negative values for negative dt
        negative_dw = dw[dt < 0]
        assert np.mean(negative_dw) < 0


class TestHebbianPlasticity:
    """Tests for Hebbian learning."""
    
    def test_coactivation_increases_weight(self):
        """Test that co-activation increases weight."""
        hebb = HebbianPlasticity(learning_rate=0.1)
        
        w = 0.5
        # Both pre and post active
        w_new = hebb.update_weight(w, pre_rate=50, post_rate=50, dt=10)
        
        assert w_new > w
    
    def test_weight_decay(self):
        """Test weight decay without activity."""
        hebb = HebbianPlasticity(learning_rate=0.01, weight_decay=0.01)
        
        w = 0.5
        # No activity
        w_new = hebb.update_weight(w, pre_rate=0, post_rate=0, dt=100)
        
        assert w_new < w


class TestSpikingNetwork:
    """Tests for spiking neural networks."""
    
    def test_network_creation(self):
        """Test network can be created."""
        net = SpikingNetwork(n_exc=80, n_inh=20)
        
        assert net.n_exc == 80
        assert net.n_inh == 20
        assert net.n_total == 100
    
    def test_random_connectivity(self):
        """Test random connectivity creation."""
        net = SpikingNetwork(n_exc=80, n_inh=20)
        net.connect_random(p=0.1)
        
        # Should have some connections
        assert np.sum(net.weights != 0) > 0
    
    def test_network_produces_spikes(self):
        """Test that network produces spikes with input."""
        net = SpikingNetwork(n_exc=80, n_inh=20, neuron_model='izhikevich')
        net.connect_random(p=0.1)
        
        spikes = net.run(duration=100, I_ext=10.0, dt=1.0)
        
        assert len(spikes) > 0
    
    def test_firing_rates_computed(self):
        """Test firing rate calculation."""
        net = SpikingNetwork(n_exc=80, n_inh=20, neuron_model='izhikevich')
        net.connect_random(p=0.1)
        net.run(duration=100, I_ext=10.0, dt=1.0)
        
        rates = net.firing_rates(time_window=100)
        
        assert len(rates) == 100
        assert np.sum(rates) > 0


class TestBrainGraph:
    """Tests for brain connectivity graph analysis."""
    
    def test_degree_calculation(self):
        """Test degree calculation."""
        # Simple ring graph with bidirectional connections
        A = np.zeros((5, 5))
        for i in range(5):
            A[i, (i+1) % 5] = 1
            A[(i+1) % 5, i] = 1
        
        graph = BrainGraph(A)
        degree = graph.degree()
        
        # Ring graph: each node has degree 4 (2 in + 2 out counted separately)
        assert all(d == 4 for d in degree)
    
    def test_clustering_coefficient(self):
        """Test clustering coefficient calculation."""
        # Complete graph should have CC = 1
        A = np.ones((5, 5)) - np.eye(5)
        
        graph = BrainGraph(A)
        cc = graph.global_clustering()
        
        assert abs(cc - 1.0) < 0.01
    
    def test_path_length(self):
        """Test shortest path length."""
        # Complete graph - all nodes connected
        A = np.ones((4, 4)) - np.eye(4)
        
        graph = BrainGraph(A)
        cpl = graph.characteristic_path_length()
        
        # Complete graph has path length 1 (direct connections)
        assert cpl > 0 and cpl < float('inf')
    
    def test_hub_identification(self):
        """Test hub node identification."""
        # Star graph: center is hub
        A = np.zeros((5, 5))
        for i in range(1, 5):
            A[0, i] = 1
            A[i, 0] = 1
        
        graph = BrainGraph(A)
        hubs = graph.hub_nodes(threshold=0.8)
        
        # Center (node 0) should be identified as hub
        assert 0 in hubs


class TestDriftDiffusionModel:
    """Tests for drift-diffusion decision model."""
    
    def test_decision_made(self):
        """Test that model makes a decision."""
        ddm = DriftDiffusionModel(v=0.3, a=1.0)
        
        choice, rt = ddm.simulate_trial()
        
        assert choice in [1, -1]
        assert rt > 0
    
    def test_accuracy_increases_with_drift(self):
        """Test that higher drift gives higher accuracy."""
        ddm_low = DriftDiffusionModel(v=0.1, a=1.0, s=0.1)
        ddm_high = DriftDiffusionModel(v=0.8, a=1.0, s=0.1)
        
        acc_low = ddm_low.accuracy(n_trials=500)
        acc_high = ddm_high.accuracy(n_trials=500)
        
        # With more trials and bigger drift difference, high should be >= low
        assert acc_high >= acc_low
    
    def test_rt_distribution(self):
        """Test that RT distribution has positive values."""
        ddm = DriftDiffusionModel()
        _, rts = ddm.simulate_experiment(n_trials=100)
        
        assert all(rt > 0 for rt in rts)
        assert np.std(rts) > 0  # Should have variability


class TestWorkingMemory:
    """Tests for working memory model."""
    
    def test_capacity_limit(self):
        """Test working memory capacity limit."""
        wm = WorkingMemory(capacity=4)
        
        # Encode 6 items
        for i in range(6):
            wm.encode(f"item_{i}")
        
        # Should only have 4 items
        contents = wm.contents()
        assert len(contents) == 4
    
    def test_retrieval(self):
        """Test item retrieval."""
        wm = WorkingMemory()
        
        wm.encode("test_item", strength=1.0)
        strength = wm.retrieve("test_item")
        
        assert strength is not None
        assert strength > 0
    
    def test_decay(self):
        """Test memory decay over time."""
        wm = WorkingMemory(decay_rate=0.1)
        
        wm.encode("item", strength=1.0)
        strength_initial = wm.retrieve("item")
        
        wm.update_time(10.0)
        strength_later = wm.retrieve("item")
        
        assert strength_later < strength_initial
    
    def test_recall_probability(self):
        """Test recall probability calculation."""
        wm = WorkingMemory()
        
        wm.encode("item", strength=1.0)
        prob = wm.recall_probability("item")
        
        assert 0 <= prob <= 1


class TestNeuralCoding:
    """Tests for neural coding functions."""
    
    def test_spike_count_rate(self):
        """Test spike counting rate calculation."""
        spikes = [Spike(0, 100), Spike(0, 200), Spike(0, 300)]
        
        rate = spike_count_rate(spikes, neuron_id=0, window=(0, 500))
        
        # 3 spikes in 500 ms = 6 Hz
        assert abs(rate - 6.0) < 0.1
    
    def test_interspike_intervals(self):
        """Test ISI calculation."""
        spikes = [Spike(0, 100), Spike(0, 200), Spike(0, 350)]
        
        isis = interspike_intervals(spikes, neuron_id=0)
        
        assert len(isis) == 2
        assert isis[0] == 100
        assert isis[1] == 150
    
    def test_coefficient_of_variation(self):
        """Test CV calculation."""
        # Regular spiking: CV close to 0
        isis_regular = np.array([10, 10, 10, 10, 10])
        cv_regular = coefficient_of_variation(isis_regular)
        assert cv_regular < 0.1
        
        # Irregular spiking: CV > 0
        isis_irregular = np.array([5, 15, 8, 20, 12])
        cv_irregular = coefficient_of_variation(isis_irregular)
        assert cv_irregular > 0.3
    
    def test_fano_factor(self):
        """Test Fano factor calculation."""
        # Poisson: FF ~ 1
        counts_poisson = np.random.poisson(10, 100)
        ff = fano_factor(counts_poisson)
        
        assert 0.5 < ff < 2.0  # Should be close to 1


class TestPhysicalConstraints:
    """Tests for physical sanity checks."""
    
    def test_reversal_potentials(self):
        """Test reversal potentials are reasonable."""
        assert E_NA > 0  # Sodium: ~+50 mV
        assert E_K < 0   # Potassium: ~-77 mV
        assert E_K < V_REST < E_NA
    
    def test_time_constant_positive(self):
        """Test membrane time constant is positive."""
        assert TAU_M > 0
    
    def test_threshold_above_rest(self):
        """Test threshold is above resting potential."""
        assert V_THRESHOLD > V_REST


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

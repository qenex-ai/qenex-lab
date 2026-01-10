"""
QENEX Neuroscience Package
==========================
Computational neuroscience covering single neurons, neural networks,
synaptic plasticity, brain connectivity, and cognitive models.
"""

from .neuroscience import (
    # Constants
    C_M, R_M, TAU_M, E_NA, E_K, E_LEAK,
    G_NA, G_K, G_LEAK, V_REST, V_THRESHOLD,
    
    # Enums
    NeuronType, SynapseType, PlasticityRule, BrainRegion,
    
    # Data classes
    Spike, SynapticConnection, NeuronState,
    
    # Neuron models
    Neuron, LeakyIntegrateAndFire, HodgkinHuxley, IzhikevichNeuron,
    
    # Plasticity
    STDP, HebbianPlasticity,
    
    # Networks
    SpikingNetwork, BrainGraph,
    
    # Neural coding functions
    spike_count_rate, interspike_intervals, coefficient_of_variation,
    fano_factor, spike_train_correlation, information_rate,
    
    # Cognitive models
    DriftDiffusionModel, WorkingMemory,
)

__version__ = "0.1.0"
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

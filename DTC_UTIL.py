
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from IPython import display
from typing import Sequence, Tuple, List, Iterator
import cirq
from cirq import Y, PhasedXZGate
#import cirq.ops.fsim_gate.PhasedFSimGate
import cirq.ops.fsim_gate as FSIM
#import cirq.PhasedFSimGate 
import cirq_google, qsimcirq
from scipy.interpolate import interp1d

class MyNoiseModel(cirq.NoiseModel):

    def __init__(self, depolarizing_error_rate, phase_damping_error_rate, amplitude_damping_error_rate):
        self._depolarizing_error_rate = depolarizing_error_rate
        self._phase_damping_error_rate = phase_damping_error_rate
        self._amplitude_damping_error_rate = amplitude_damping_error_rate

    def noisy_operation(self, op):
        n_qubits = len(op.qubits)
        depolarize_channel = cirq.depolarize(self._depolarizing_error_rate, n_qubits=n_qubits)
        phase_damping_channel = cirq.phase_damp(self._phase_damping_error_rate).on_each(op.qubits)
        amplitude_damping_channel = cirq.amplitude_damp(self._amplitude_damping_error_rate).on_each(op.qubits)
        return [op, depolarize_channel.on(*op.qubits), phase_damping_channel, amplitude_damping_channel]


def simulate_dtc_circuit_list(
                            circuit_list: Sequence[cirq.Circuit],
                            noise_md: cirq.NoiseModel
                            ) -> np.ndarray:

    simulator = cirq.Simulator(noise=noise_md)
    circuit_positions = {len(c) - 1 for c in circuit_list}
    circuit = circuit_list[-1]

    probabilities = []
    for k, step in enumerate(
        simulator.simulate_moment_steps(circuit=circuit)
    ):
        # add the state vector if the number of moments simulated so far is equal
        #   to the length of a circuit in the circuit_list
        if k in circuit_positions:
            probabilities.append(np.abs(step.state_vector()) ** 2)

    return np.asarray(probabilities)

def get_polarizations(
    probabilities: np.ndarray,
    num_qubits: int,
    initial_states: np.ndarray = None,
) -> np.ndarray:
    """Get polarizations from matrix of probabilities, possibly autocorrelated on
        the initial state.

    A polarization is the marginal probability for a qubit to measure zero or one,
        over all possible basis states, scaled to the range [-1. 1].

    Args:
        probabilities: `np.ndarray` of shape (:, cycles, 2**qubits)
            representing probability to measure each bit string
        num_qubits: the number of qubits in the circuit the probabilities
            were generated from
        initial_states: `np.ndarray` of shape (:, qubits) representing the initial
            state for each dtc circuit list

    Returns:
        `np.ndarray` of shape (:, cycles, qubits) that represents each
            qubit's polarization

    """
    # prepare list of polarizations for each qubit
    polarizations = []
    for qubit_index in range(num_qubits):
        # select all indices in range(2**num_qubits) for which the
        #   associated element of the statevector has qubit_index as zero
        shift_by = num_qubits - qubit_index - 1
        state_vector_indices = [
            i for i in range(2**num_qubits) if not (i >> shift_by) % 2
        ]

        # sum over all probabilities for qubit states for which qubit_index is zero,
        #   and rescale them to [-1,1]
        polarization = (
            2.0
            * np.sum(
                probabilities.take(indices=state_vector_indices, axis=-1),
                axis=-1,
            )
            - 1.0
        )
        polarizations.append(polarization)

    # turn polarizations list into an array,
    #   and move the new, leftmost axis for qubits to the end
    polarizations = np.moveaxis(np.asarray(polarizations), 0, -1)

    # flip polarizations according to the associated initial_state, if provided
    #   this means that the polarization of a qubit is relative to it's initial state
    if initial_states is not None:
        initial_states = 1 - 2.0 * initial_states
        polarizations = initial_states * polarizations

    return polarizations
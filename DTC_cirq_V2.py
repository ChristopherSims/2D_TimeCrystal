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
from DTC_UTIL import *


def DTC_Circuit_list(
                    qubits, 
                    cycles: int,
                    g_value: float,
                    theta: float,
                    phi: float,
                    alpha: float,
                    beta: float,
                    local_fields: float,
                    ) -> List[cirq.Circuit]:
    circuit = cirq.Circuit()
    ## initial operation
    initial_operations = []
    for index, qubit in enumerate(qubits):
      initial_operations.append(Y(qubit))  
    initial_operations = cirq.Moment(initial_operations)
    ## Initial U gate
    sequence_operations = []
    for index, qubit in enumerate(qubits):
        sequence_operations.append(
            cirq.PhasedXZGate(
                x_exponent=g_value,
                axis_phase_exponent=0.0,
                z_exponent=local_fields,
            )(qubit)
        )
    u_cycle = [cirq.Moment(sequence_operations)]

    #FSIM Gate
    even_qubit_moment = []
    odd_qubit_moment = []
    for index, (qubit, next_qubit) in enumerate(zip(qubits, qubits[1:])):
        # Add an fsim gate
        coupling_gate = cirq.ops.PhasedFSimGate.from_fsim_rz(
            theta=theta,
            phi = phi,
            rz_angles_before = (alpha,alpha),
            rz_angles_after = (beta,beta)
        )

        if index % 2:
            even_qubit_moment.append(coupling_gate.on(qubit, next_qubit))
        else:
            odd_qubit_moment.append(coupling_gate.on(qubit, next_qubit))
    u_cycle.append(cirq.Moment(even_qubit_moment))
    u_cycle.append(cirq.Moment(odd_qubit_moment))

    circuit_list = []
    total_circuit = cirq.Circuit(initial_operations)
    circuit_list.append(total_circuit.copy())
    for _ in range(cycles):
        for moment in u_cycle:
            total_circuit.append(moment)
        circuit_list.append(total_circuit.copy())

    return circuit_list



def DTC_Circuit_list_2D(
                    sq:int,
                    qubits, 
                    cycles: int,
                    g_value: float,
                    theta: float,
                    phi: float,
                    alpha: float,
                    beta: float,
                    local_fields: float,
                    ) -> List[cirq.Circuit]:

    circuit = cirq.Circuit()
    ## initial operation
    initial_operations = []
    for index, qubit in enumerate(qubits):
      initial_operations.append(Y(qubit))  
    initial_operations = cirq.Moment(initial_operations)
    ## Initial U gate
    sequence_operations = []
    for index, qubit in enumerate(qubits):
        sequence_operations.append(
            cirq.PhasedXZGate(
                x_exponent=g_value,
                axis_phase_exponent=0.0,
                z_exponent=local_fields,
            )(qubit)
        )
    u_cycle = [cirq.Moment(sequence_operations)]

    #FSIM Gate
    even_qubit_moment = []
    odd_qubit_moment = []

    for i in range(0, sq-1, 2):
        for j in range(0,sq-1,2):
            coupling_gate = cirq.ops.PhasedFSimGate.from_fsim_rz(
            theta=theta,
            phi = phi,
            rz_angles_before = (alpha,alpha),
            rz_angles_after = (beta,beta)
            )
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*(i+1) + j])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*(i+1) + j+1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*i + j+1])))
         
    for i in range(1, sq-1, 2):
        for j in range(1,sq-1,2):
            coupling_gate = cirq.ops.PhasedFSimGate.from_fsim_rz(
            theta=theta,
            phi = phi,
            rz_angles_before = (alpha,alpha),
            rz_angles_after = (beta,beta)
            )
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*(i+1) + j])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*(i+1) + j+1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*i + j+1])))


    circuit_list = []
    total_circuit = cirq.Circuit(initial_operations)
    circuit_list.append(total_circuit.copy())
    for _ in range(cycles):
        for moment in u_cycle:
            total_circuit.append(moment)
        circuit_list.append(total_circuit.copy())

    return circuit_list


# Define the error rates
depolarizing_error_rate = 0.00
phase_damping_error_rate = 0.00
amplitude_damping_error_rate = 0.00
my_noise_model = MyNoiseModel(depolarizing_error_rate, phase_damping_error_rate, amplitude_damping_error_rate)



sq = 4
N_QUBITS = sq**2 #Number of qubits used in Google paper

qubits = cirq.LineQubit.range(N_QUBITS)
qubits = cirq.GridQubit.square(sq)
num_cycles = 50
D = 0.001
G = 0.9
floq_circuit = DTC_Circuit_list_2D(
                    sq,
                    qubits, 
                    cycles = num_cycles,
                    g_value = G,
                    theta = 1*np.pi,
                    phi = 1*np.pi,
                    alpha = 1*np.pi,
                    beta = 1*np.pi,
                    local_fields = D
                    )
floq_circuit = DTC_Circuit_list(
                    qubits, 
                    cycles = num_cycles,
                    g_value = G,
                    theta = 1*np.pi,
                    phi = 1*np.pi,
                    alpha = 1*np.pi,
                    beta = 1*np.pi,
                    local_fields = D
                    )

                    





# result = simulate_dtc_circuit_list(floq_circuit,noise_md=my_noise_model)

# dtc_z = np.transpose(get_polarizations(result, N_QUBITS))
# print(dtc_z.shape)
# fig = plt.figure(figsize=(12,8))
# ax = plt.subplot(1,1,1)
# im = ax.imshow(dtc_z)
# plt.rcParams.update({'font.size': 15})
# #plt.rcParams['text.usetex'] = True

# # Graph results
# ax.set_xlabel('Floquet time step',fontsize=24)
# ax.xaxis.labelpad = 10
# ax.set_ylabel('Qubit',fontsize=24)
# ax.set_xticks(np.append(np.arange(0, num_cycles , 10),num_cycles),fontsize=18)
# yticks_new = np.arange(0, N_QUBITS, sq)
# ax.set_yticks(yticks_new)
# ax.set_yticklabels(yticks_new+1,fontsize=18)
# ax.tick_params(axis='both', which='major', labelsize=24)
# plt.ylim([N_QUBITS-0.5,-0.5])
# ax.xaxis.set_label_position('bottom')
# plt.savefig('1D_POL_G%0.2f_D%0.2f.png'%(G,D),bbox_inches='tight')
# plt.show()


# fig, ax = plt.subplots(int(N_QUBITS/2),2)
# #ax = plt.subplot(1,1,1)
# #im = ax.imshow(dtc_z)
# dtc_b = dtc_z - dtc_z.mean(axis=1, keepdims=True)
# for ii in range(int(N_QUBITS/2)):
#     for jj in range(2):
#         ax[int(ii),int(jj)].plot(dtc_b[int(ii)+int(jj),:], linestyle = '--', marker = 'o')
#         ax[int(ii),int(jj)].set_ylim([-1,1])
#         ax[int(ii),int(jj)].set_ylim([-1,1])
#         ax[int(ii),int(jj)].set_xticks([])
#         if ii % 2 == 1:
#             ax[int(ii),int(jj)].set_yticks([])
# plt.rcParams.update({'font.size': 12})
# #plt.rcParams['text.usetex'] = True
# DTC_sum = np.sum(dtc_b,axis=0)/N_QUBITS
# #np.save('DTC_SUM_G%0.2f_D%0.2f_1D'%(G,D),DTC_sum)
# print(DTC_sum.shape)
# # # Graph results
# # ax.set_xlabel('Floquet cycles (t)')
# # ax.xaxis.labelpad = 10
# # ax.set_ylabel('Z')
# # ax.set_xticks(np.arange(0, num_cycles , 10))
# # #ax.set_yticks(np.arange(0, N_QUBITS, 5))
# # ax.xaxis.set_label_position('top')
# #plt.show()
# fig = plt.figure()
# ax = plt.subplot(1,1,1)
# ax.plot(DTC_sum,linewidth = 3,marker = 'o')
# plt.axline((0,0),(num_cycles,0),color='k',linewidth = 2,linestyle='--')
# ax.set_ylabel(r'$\overline{ \langle Z(0)Z(t) \rangle}$',fontsize=24)
# ax.set_xlabel('Floquet time step',fontsize=24)
# ax.tick_params(axis='both', which='major', labelsize=18)
# ax.set_yticks([1.00,0.5,0.0,-0.5,-1.0],minor=True,fontsize=18)
# ax.minorticks_on()
# #ax.yaxis.set_minor_locator(MultipleLocator(5))
# ax.tick_params(which='major', length=10, width=2, direction='in',labelsize=18)
# ax.tick_params(which='minor', length=5, width=2, direction='in',labelsize=18)
# #plt.show()


# fig = plt.figure()
# ax = plt.subplot(1,1,1)
# f_s = 51
# samplingFrequency   = 1
# samplingInterval       = 1 / samplingFrequency
# fourierTransform = np.fft.fft(DTC_sum)/len(DTC_sum)
# fourierTransform = fourierTransform[range(int(len(DTC_sum)))]
# tpCount     = len(DTC_sum)
# values      = np.arange(int(tpCount))
# timePeriod  = tpCount/samplingFrequency
# frequencies = values/timePeriod
# plt.plot(frequencies, np.abs(fourierTransform),marker='o',linewidth=4)
# ax.minorticks_on()
# #ax.tick_params(axis='both', which='major', labelsize=18)
# ax.tick_params(which='major', length=10, width=2, direction='in',labelsize=18)
# ax.tick_params(which='minor', length=5, width=2, direction='in',labelsize=18)
# ax.set_ylabel(r'$\mathscr{F} \{\overline{ \langle Z(0)Z(t) \rangle} \}$',fontsize=24)
# ax.set_xlabel(r'$\omega /\omega_D$',fontsize=24)
# plt.ylim([-0.01,0.51])
# #plt.show()
# #plt.plot(dtc_z[1,:])



import numpy as np
from qiskit_aer import Aer
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit import transpile
import matplotlib.pyplot as plt 
from qiskit.visualization import plot_histogram

np.random.seed(42)  
num_jobs = 4
num_machines = 3
num_slots = num_jobs  

operations = np.random.randint(0, num_machines, size=(num_jobs, num_machines))
print(f"Casual assignment job to machine:\n {operations}")


num_qubits = num_jobs * num_slots

qubo_terms = []
qubo_weights = []

# H1 = -SUM(Xj,t) indicate job j can be execute at slot t 
# 1 job assign at 1 slot t 
for j in range(num_jobs):
    for t in range(num_slots):
        qubit_idx = j * num_slots + t # idx job j at time t 
        qubo_terms.append([qubit_idx])  
        qubo_weights.append(-1)  

# H2 = SUM SUM (2Xj1,t Xj2,t)
# Xjt indicate if job j assign slot t 
# if 2 job assign same slot cost increase 
for m in range(num_machines):
    for t in range(num_slots):
        for j1 in range(num_jobs): # fist job 
            for j2 in range(j1 + 1, num_jobs): # second job where j2 > j1
                q1 = j1 * num_slots + t # qb1 job j1 at time t
                q2 = j2 * num_slots + t # qb2 job j2 at time t 
                qubo_terms.append([q1, q2]) 
                qubo_weights.append(2)  # penalty for overlapping assignment 

qubo_terms_fixed = []
for term in qubo_terms:
    pauli_string = ['I'] * num_qubits  # init I string with all qubit 
    for index in term:                 
        pauli_string[index] = 'Z'      # change I with Z qubit involved in a constraint
    qubo_terms_fixed.append(''.join(pauli_string))

hamiltonian = SparsePauliOp.from_list(list(zip(qubo_terms_fixed, qubo_weights)))

ansatz = EfficientSU2(num_qubits=num_qubits)

# find parameter antsaz calculate the quantistic state
simulator = Aer.get_backend('aer_simulator_statevector')

optimizer = COBYLA(maxiter=1000)

estimator = Estimator()

vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)

result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

print(f"Min energy found: {result.eigenvalue.real}")
for param, value in result.optimal_parameters.items():
    print(f"{param}: {value:.4f}") 


# assign optimal parameter to antsaz 
optimal_circuit = ansatz.assign_parameters(result.optimal_parameters)

optimal_circuit.measure_all()

# using measure for find solution
simulator = Aer.get_backend('aer_simulator')

transpiled_circuit = transpile(optimal_circuit, simulator)

job = simulator.run(transpiled_circuit, shots=2048)  
result_sim = job.result()
counts = result_sim.get_counts() 

sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10])

print(sorted_counts)
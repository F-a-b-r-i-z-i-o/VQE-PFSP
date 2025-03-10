import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_aer import Aer
from qiskit import transpile
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.optimizers import COBYLA

num_jobs = 5
num_machines = 2

# Matrix of cost_matrix[j][m] = assign Job j a Machine m
cost_matrix = np.random.randint(1, 10, size=(num_jobs, num_machines))
print("Matrix of cost (Job x Machine):\n", cost_matrix)

qp = QuadraticProgram()

# Added  binary variable x_{j,m}: 1 if Job j is on Machine m, 0 othrewise
for j in range(num_jobs):
    for m in range(num_machines):
        qp.binary_var(name=f"x_{j}_{m}")

# Objj fun minimize sum of cost cost_matrix[j,m]*x_{j,m}
linear_obj = {}
for j in range(num_jobs):
    for m in range(num_machines):
        linear_obj[f"x_{j}_{m}"] = cost_matrix[j, m]
qp.minimize(linear=linear_obj)

# Every job assign 1 machine
for j in range(num_jobs):
    qp.linear_constraint(
        {f"x_{j}_{m}": 1 for m in range(num_machines)},
        sense='==',
        rhs=1
    )


conv = QuadraticProgramToQubo()
qubo = conv.convert(qp)
operator, offset = qubo.to_ising()

ansatz = EfficientSU2(operator.num_qubits)

optimizer = COBYLA(maxiter=1000)

vqe = VQE(estimator=Estimator(), ansatz=ansatz, optimizer=optimizer)

print("\nExecution VQE...")
result_vqe = vqe.compute_minimum_eigenvalue(operator)

print(f"\nMin energy found: {result_vqe.eigenvalue.real:.4f}")
print("Best parameter found:")
for i, (param, val) in enumerate(result_vqe.optimal_parameters.items()):
    print(f"{param}: {val:.4f}")

optimal_circuit = ansatz.assign_parameters(result_vqe.optimal_parameters)
optimal_circuit.measure_all()

backend = Aer.get_backend('aer_simulator')
transpiled_circuit = transpile(optimal_circuit, backend)
shots = 1024
job = backend.run(transpiled_circuit, shots=shots)
result_sim = job.result()
counts = result_sim.get_counts()


# Order ascending
sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

print("\n 10 bitstring most frequent:")
for i, (bitstring, cnt) in enumerate(sorted_counts.items()):
    if i < 10:
        print(f"{bitstring} -> {cnt} occurrences")
    else:
        break

best_bitstring = next(iter(sorted_counts))  # most frequent
bit_list = list(reversed(best_bitstring))  # reverse for qiskit order
print(f"\n Bitstring best (most frequent): {best_bitstring}")

assignments = [[] for _ in range(num_jobs)]
idx = 0
for j in range(num_jobs):
    for m in range(num_machines):
        if bit_list[idx] == '1':
            assignments[j].append(m)
        idx += 1

print("\n Assignment found bitstring (Job -> List of Machine):")
for j in range(num_jobs):
    print(f"Job {j} assign at machine {assignments[j]}")

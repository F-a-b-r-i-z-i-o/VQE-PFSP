from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import SPSA, COBYLA
import itertools
import numpy as np
from scipy.optimize import minimize
import math

simulator = StatevectorSampler()

class PFSProblem:
    def __init__(self, ptime):
        self.nj, self.nm = ptime.shape
        self.ptime = ptime

    @staticmethod
    def load(fname):
        with open(fname, "r") as f:
            f.readline()
            nj, nm = [int(x) for x in f.readline().strip().split()][:2]
            f.readline()
            ptime = np.zeros((nj, nm), dtype=np.int32)
            for m in range(nm):
                ptime[:, m] = [int(x) for x in f.readline().strip().split()]
        return PFSProblem(ptime)

    @staticmethod
    def int2perm(x, n):
        l = list(range(n))
        p = []
        while n >= 1:
            r = x % n
            x //= n
            p.append(l[r])
            del l[r]
            n -= 1
        return p

    def evaluate(self, p):
        ctime = np.zeros((self.nj, self.nm), dtype=np.int32)
        f = p[0]
        ctime[f, 0] = self.ptime[f, 0]
        for i in range(1, self.nm):
            ctime[f, i] = ctime[f, i - 1] + self.ptime[f, i]
        ms = ctime[f, self.nm - 1]
        h = f
        for j in range(1, self.nj):
            k = p[j]
            ctime[k, 0] = ctime[h, 0] + self.ptime[k, 0]
            for i in range(1, self.nm):
                ctime[k, i] = max(ctime[k, i - 1], ctime[h, i]) + self.ptime[k, i]
            ms = max(ms, ctime[k, self.nm - 1])
            h = k
        return ms

    def evaluateb(self, x):
        k = PFSProblem.int2perm(int(x, 2), self.nj)
        return self.evaluate(k)

    def find_optima(self):
        fmin = self.ptime.sum() + 1
        opt = []
        for c in itertools.permutations(range(self.nj)):
            f = self.evaluate(c)
            if f < fmin:
                fmin = f
                opt = [c]
            elif f == fmin:
                opt.append(c)
        return fmin, opt

class vqe_pfsp:
    def __init__(self, prob, nj):
        self.problem = prob
        self.problem.nj = nj

    def create_ansatz(self, ans="Real", reps=1):
        fact = np.prod(range(1, self.problem.nj + 1))
        nqb = int(0.5 + np.log2(fact))
        self.ansatz = RealAmplitudes(nqb, reps=reps)
        self.npar = self.ansatz.num_parameters
        print(f"created ansatz with {nqb} qubits and {self.npar} parameters")
        self.ansatz.measure_all()

    def find_greedy_classical_solution(self):
        return list(range(self.problem.nj))

    def classical_solution_to_params(self, permutation):
        fact = math.factorial(self.problem.nj)
        nqubits = int(np.ceil(np.log2(fact)))
        permutation_index = None
        for idx, perm in enumerate(itertools.permutations(range(self.problem.nj))):
            if list(perm) == permutation:
                permutation_index = idx
                break
        if permutation_index is None:
            raise ValueError("Invalid permutation provided")

        bin_index = format(permutation_index, f'0{nqubits}b')

        reps = self.ansatz.reps
        skip_final = self.ansatz._skip_final_rotation_layer
        n_blocks = reps + (0 if skip_final else 1)

        theta_init = np.zeros(self.ansatz.num_parameters)

        idx = 0
        for block in range(n_blocks):
            for q in range(nqubits):
                theta_init[idx] = 0.0 if bin_index[q] == '1' else np.pi
                idx += 1
        return theta_init


    def objf_avg(self, params):
        counts = self.simulate(params)
        total = sum(freq * self.problem.evaluateb(bs[::-1]) for bs, freq in counts.items())
        return total / 1024

    def simulate(self, params, nshots=1024):
        circuit = self.ansatz.assign_parameters(params)
        job = simulator.run([(circuit,)], shots=nshots)
        res = job.result()[0]
        counts = res.data.meas.get_counts()
        return counts

    def run(self, num_tries=10, max_iter_spa=500, max_iter_cobyla=500):
        fmin, opt = self.problem.find_optima()
        print(f"minimum {fmin}, number of optima {len(opt)}")

        samples = [
            (s, self.objf_avg(s)) for s in (
                2 * np.pi * np.random.random(self.npar) for _ in range(num_tries**2)
            )
        ]

        x_classic = self.find_greedy_classical_solution()
        ws_params = self.classical_solution_to_params(x_classic)
        cost_ws = self.objf_avg(ws_params)
        samples.append((ws_params, cost_ws))

        samples.sort(key=lambda c: c[1])

        max_num_success = 0
        best_energy = np.inf

        for r in range(num_tries):
            x0 = samples[r][0]
            init_fun = samples[r][1]

            res_spsa = SPSA(maxiter=max_iter_spa).minimize(fun=self.objf_avg, x0=x0)
            res_cobyla = COBYLA(maxiter=max_iter_cobyla).minimize(fun=self.objf_avg, x0=res_spsa.x)

            counts = self.simulate(res_cobyla.x)
            num_success = sum(freq * (self.problem.evaluateb(bs[::-1]) == fmin) for bs, freq in counts.items())

            print(f"{r+1} Initial {init_fun:.3f}, final average {res_cobyla.fun:.3f} Prob. opt. {num_success/1024:.3f}")

            if num_success > max_num_success:
                max_num_success = num_success
                best_energy = res_cobyla.fun

        print(f"Best result average energy {best_energy:.3f} Prob. opt. {max_num_success/1024:.3f}")



def run():
    prob = PFSProblem.load("tai20_5_0.fsp")
    vqe = vqe_pfsp(prob, 4)
    vqe.create_ansatz("EffSU2", 1)
    vqe.run()

if __name__ == "__main__":
    run()

from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import ADAM, SPSA, COBYLA
import numpy as np
import os 
import csv
from itertools import product 

simulator = StatevectorSampler()

class FossProblem:
    def __init__(self, ptime):
        self.nj, self.nm = ptime.shape
        self.ptime = ptime

    @staticmethod
    def generate_random(nj, nm, maxt, seed=None):
        if seed:
            np.random.seed(seed)
        ptime = np.random.randint(1, maxt, size=(nj, nm))
        return FossProblem(ptime)

    def evaluateb(self, x):
        ctime = np.zeros(self.nm)
        for i in range(self.nj):
            ctime[int(x[i])] += self.ptime[i, int(x[i])]
        return max(ctime)

    def find_optima(self):
        c = np.zeros(self.nj, dtype=int)
        fmin = np.inf
        opt = []
        for _ in range(self.nm ** self.nj):
            f = self.evaluateb(c)
            if f < fmin:
                fmin = f
                opt = [c.copy()]
            elif f == fmin:
                opt.append(c.copy())
            j = 0
            while j < self.nj:
                c[j] += 1
                if c[j] < self.nm:
                    break
                c[j] = 0
                j += 1
        return fmin, opt

class vqe_foss:
    def __init__(self, prob):
        self.problem = prob
        self.simulator = StatevectorSampler()
        self.n_shot = 1024
        self.ansatz = None
        self.npar = None

    def create_custom_ansatz(self, reps=3):
        self.ansatz = RealAmplitudes(self.problem.nj*self.problem.nm, reps=reps, entanglement='linear')
        self.ansatz.measure_all()
        self.npar = self.ansatz.num_parameters

    def objf_avg(self, params):
        counts = self.simulate(params)
        total = sum(freq * self.problem.evaluateb(bs[::-1]) for bs, freq in counts.items())
        return total / self.n_shot

    def simulate(self, params, nshots=1024):
        circuit = self.ansatz.assign_parameters(params)
        job = self.simulator.run([(circuit,)], shots=nshots)
        res = job.result()[0]
        counts = res.data.meas.get_counts()
        return counts
    
    def find_greedy_classical_solution(self):
        """
        Assign job to finish machine 
        """
        ctime = np.zeros(self.problem.nm)
        x = np.zeros(self.problem.nj, dtype=int)
        for j in range(self.problem.nj):
            m = np.argmin(ctime)
            x[j] = m
            ctime[m] += self.problem.ptime[j, m]
    
        return x

    def classical_solution_to_params(self, x):
        reps = self.ansatz.reps                    
        skip_final = self.ansatz._skip_final_rotation_layer  
        n_blocks = reps + (0 if skip_final == True else 1)  

        npar = self.ansatz.num_parameters 
        theta_init = np.zeros(npar)

        idx = 0
        for block in range(n_blocks):
            for job in range(self.problem.nj):
                chosen_machine = x[job]
                for machine in range(self.problem.nm):
                    if machine == chosen_machine:
                        theta_init[idx] = 0.0  # state |1> for select macchine
                    else:
                        theta_init[idx] = np.pi  # state |0> for non select machine 
                    idx += 1

        return theta_init


    def run(self, num_tries=5, max_iter_spsa=100, max_iter_cobyla=1000, csv_filename='result.csv'):

        fmin, opt = self.problem.find_optima()
        print(f"Minimum makespan {fmin}, number of optimal solutions {len(opt)}")

        
        samples = [(2 * np.pi * np.random.random(self.npar)) for _ in range(num_tries**2)]
        evaluated_samples = [(s, self.objf_avg(s)) for s in samples]

        
        x_classic = self.find_greedy_classical_solution()
        ws_params = self.classical_solution_to_params(x_classic)
        cost_ws = self.objf_avg(ws_params)
        evaluated_samples.append((ws_params, cost_ws))

        evaluated_samples.sort(key=lambda x: x[1])
        
        os.makedirs('result_vqe', exist_ok=True)
        csv_file_path = os.path.join('result_vqe', csv_filename)

        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Attempt', 'Initial Cost', 'SPSA Cost', 'COBYLA Cost', 'Probability Optimal'])

            for r in range(num_tries):
                x0, init_fun = evaluated_samples[r]

                print(f"\n=== Attempt {r+1} ===")
                print(f"Starting from cost: {init_fun:.3f}")

                print("Call SPSA for GLOBAL OPT")
                optimizer_spsa = SPSA(maxiter=max_iter_spsa)
                res_spsa = optimizer_spsa.minimize(fun=self.objf_avg, x0=x0)

                print("Call COBYLA for LOCAL OPT")
                optimizer_cobyla = COBYLA(maxiter=max_iter_cobyla)
                res_cobyla = optimizer_cobyla.minimize(fun=self.objf_avg, x0=res_spsa.x)

                counts = self.simulate(res_cobyla.x)
                num_success = sum(freq for bs, freq in counts.items()if self.problem.evaluateb(bs[::-1]) == fmin)
                prob_optimal = num_success / self.n_shot

                csv_writer.writerow([r + 1, f"{init_fun:.3f}", f"{res_spsa.fun:.3f}", f"{res_cobyla.fun:.3f}", f"{prob_optimal:.3f}"])

                print(f"Results - SPSA cost: {res_spsa.fun:.3f}, "f"COBYLA cost: {res_cobyla.fun:.3f}, "f"Prob. optimal: {prob_optimal:.3f}")



if __name__ == "__main__":
    num_jobs = [4, 5, 6, 7]
    num_machines = [2, 3]
    maxt = 10

    for nj, nm in product(num_jobs, num_machines):
        print(f"\n--- Running with {nj} jobs and {nm} machines ---")
        problem = FossProblem.generate_random(nj, nm, maxt)
        solver = vqe_foss(problem)
        solver.create_custom_ansatz()
        solver.run(csv_filename=f'results_{nj}jobs_{nm}machines.csv')
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import SPSA, COBYLA
import itertools
import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd 
import os 
import glob

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

    def create_ansatz(self, reps=1):
        fact = np.prod(range(1, self.problem.nj + 1))
        nqb = int(0.99 + np.log2(fact))
        self.ansatz = RealAmplitudes(nqb, reps=reps)
        self.npar = self.ansatz.num_parameters
        print(f"created ansatz with {nqb} qubits and {self.npar} parameters")
        self.ansatz.measure_all()

    def find_greedy_classical_solution(self):

        # Estimate total time of elaboration for each job 
        job_total_times = [(job, sum(self.problem.ptime[job])) for job in range(self.problem.nj)]
        
        # Order job in ascending order by total time  ( most longe first )
        greedy_permutation = [job for job, _ in sorted(job_total_times, key=lambda x: -x[1])]
      
        return greedy_permutation

    def classical_solution_to_params(self, permutation):
        perturbation = 0.05

        # Calculate the total number of possible permutations (nj!).
        fact = math.factorial(self.problem.nj)
        # Calculate the number of qubits needed to represent all permutations.
        nqubits = int(np.ceil(np.log2(fact)))
        
        permutation_index = None
        # Find the index of the given permutation with respect to all possible permutations
        for idx, perm in enumerate(itertools.permutations(range(self.problem.nj))):
            if list(perm) == permutation:
                permutation_index = idx
                break
        
        # Converts the index of the permutation to a binary string.
        bin_index = format(permutation_index, f'0{nqubits}b')

        theta_init = np.zeros(self.ansatz.num_parameters)

        for idx in range(self.ansatz.num_parameters):
            # Use idx module nqubits to avoid out-of-bound index errors
            bit_value = bin_index[idx % nqubits]
            
            if bit_value == '1':
                theta_init[idx] = np.random.uniform(0, perturbation)
            else:
                theta_init[idx] = np.pi + np.random.uniform(0, perturbation)

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

    def run(self, num_tries=10, max_iter_spa=1000, max_iter_cobyla=1000, instance_name="", run_id=0):
      
        fmin, opt_solutions = self.problem.find_optima()
        print(f"Theoretical min value (makespan): {fmin}")

       
        samples = []
        for _ in range(num_tries**2):
            x_rand = 2 * np.pi * np.random.random(self.npar)
            samples.append((x_rand, self.objf_avg(x_rand)))
        
        # x_classic = self.find_greedy_classical_solution()
        # ws_params = self.classical_solution_to_params(x_classic)
        # cost_ws = self.objf_avg(ws_params)
        # samples.append((ws_params, cost_ws))
        
        samples.sort(key=lambda c: c[1])

        log_results = []

        for r in range(num_tries):
            x0 = samples[r][0]
            init_fun = samples[r][1]

            # SPSA
            res_spsa = SPSA(maxiter=max_iter_spa).minimize(fun=self.objf_avg, x0=x0)

            # COBYLA
            res_cobyla = COBYLA(maxiter=max_iter_cobyla).minimize(fun=self.objf_avg, x0=res_spsa.x)
            
            counts = self.simulate(res_cobyla.x)
            nshots = 1024
            
            # calcolate probability of find fmin 
            num_success = sum(
                freq for bs, freq in counts.items() 
                if self.problem.evaluateb(bs[::-1]) == fmin
            )
            prob_opt = num_success / nshots
           
            log_results.append({
                "instance": instance_name,
                "run": run_id,
                "n_job": self.problem.nj,
                "n_tries": r + 1,
                "e_min": fmin,
                "initial_average": init_fun,
                "final_average": res_cobyla.fun,
                "prob_opt": prob_opt
            })
            

        return log_results

    
    def run_experiments(instance_pattern="tai20_5_*.fsp", jobs_list=[4, 5, 6], runs_per_instance=3):
        all_logs = []
        os.makedirs("results", exist_ok=True)

        for nj in jobs_list:
            for instance_file in sorted(glob.glob(instance_pattern)):
                instance_name = os.path.basename(instance_file)
                print(f"Processing instance: {instance_name}, Jobs: {nj}")

                prob = PFSProblem.load(instance_file)
                vqe = vqe_pfsp(prob, nj)
                vqe.create_ansatz(reps=1)

                for run_id in range(1, runs_per_instance + 1):
                    print(f"Running instance: {instance_name}, Jobs: {nj}, Run: {run_id}")
                    logs = vqe.run(
                        num_tries=10,
                        max_iter_spa=1000,
                        max_iter_cobyla=1000,
                        instance_name=instance_name,
                        run_id=run_id
                    )
                    all_logs.extend(logs)

                    df = pd.DataFrame(all_logs)
                    csv_path = os.path.join("results", "results_vqe_pfsp.csv")
                    df.to_csv(csv_path, index=False)
        

if __name__ == "__main__":
    vqe_pfsp.run_experiments("taillard/tai20_5_*.fsp", jobs_list=[2, 3, 4, 5, 6], runs_per_instance=5)


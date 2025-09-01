from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import SPSA, COBYLA
from src.problem import PFSProblem
import numpy as np
import pandas as pd 
import os 
import glob

simulator = StatevectorSampler()

class VQE:
    """
    Variational algorithm for PFSP using a RealAmplitudes ansatz.

    The ansatz produces a probability distribution over bitstrings.
    Each bitstring (reversed) is decoded into a job permutation, which is
    evaluated by the PFSP makespan. 
    The objective is the empirical average makespan over measurement outcomes.
    """

    def __init__(self, prob: PFSProblem, nj: int) -> None:
        """Initialize with a PFSP instance and the number of jobs considered.

        Parameters
        ----------
        prob : PFSProblem
            Instance containing processing times.
        nj : int
            Number of jobs to optimize (overrides prob.nj for this run).

        Returns
        -------
        None
        """
        self.problem = prob
        self.problem.nj = nj

    def create_ansatz(self, reps: int = 1) -> None:
        """
        Create a RealAmplitudes ansatz sized for ≈log2(nj!) qubits and measure all.

        Parameters
        ----------
        reps : int
            Number of repetition layers in the hardware-efficient ansatz.

        Returns
        -------
        None
        """
        fact = np.prod(range(1, self.problem.nj + 1))
        nqb = int(0.99 + np.log2(fact))
        self.ansatz = RealAmplitudes(nqb, reps=reps)
        self.npar = self.ansatz.num_parameters
        print(f"created ansatz with {nqb} qubits and {self.npar} parameters")
        # Measure all qubits so simulate() returns classical bitstring counts
        self.ansatz.measure_all()

    def objf_avg(self, params: np.ndarray) -> float:
        """
        Objective: average makespan under the circuit's measurement distribution.

        Parameters
        ----------
        params : np.ndarray
            Ansatz parameters to bind.

        Returns
        -------
        float
            Empirical average makespan, computed from counts and divided by 1024.
            (The denominator mirrors the fixed nshots used elsewhere in this code.)
        """
        counts = self.simulate(params)
        # Map each measured bitstring (reversed) to a permutation and its makespan
        total = sum(freq * self.problem.evaluateb(bs[::-1]) for bs, freq in counts.items())
        return total / 1024

    def simulate(self, params: np.ndarray, nshots: int = 1024) -> Dict[str, int]:
        """
        Simulate the parameterized circuit and return measurement counts.

        Parameters
        ----------
        params : np.ndarray
            Ansatz parameters to assign.
        nshots : int
            Number of shots for sampling (kept at 1024 by default, as assumed elsewhere).

        Returns
        -------
        dict[str, int]
            Mapping bitstring -> observed frequency.
        """
        # Bind parameters and execute with the shared StatevectorSampler
        circuit = self.ansatz.assign_parameters(params)
        job = simulator.run([(circuit,)], shots=nshots)
        res = job.result()[0]
        # Assumes a result object exposing .data.meas.get_counts()
        counts = res.data.meas.get_counts()
        return counts

    def run(
        self,
        num_tries: int = 10,
        max_iter_spa: int = 1000,
        max_iter_cobyla: int = 1000,
        instance_name: str = "",
        run_id: int = 0
    ) -> list[dict]:
        """
        Perform optimization: random seeds → SPSA → COBYLA

        Parameters
        ----------
        num_tries : int
            Number of restarts (best seeds kept from a larger random pool).
        max_iter_spa : int
            Max iterations for SPSA.
        max_iter_cobyla : int
            Max iterations for COBYLA.
        instance_name : str
            Label stored in the log.
        run_id : int
            Sequential run identifier for the same instance.

        Returns
        -------
        list[dict]
            Metrics for each try (initial/final averages, prob. of optimum, etc.).
        """
        # 1) Exact optimum for reference
        fmin, opt_solutions = self.problem.find_optima()
        print(f"Theoretical min value (makespan): {fmin}")

        # 2) Randomly sample many starts and evaluate their average objective
        samples = []
        for _ in range(num_tries**2):
            x_rand = 2 * np.pi * np.random.random(self.npar)
            samples.append((x_rand, self.objf_avg(x_rand)))
        
        # Keep the best num_tries seeds
        samples.sort(key=lambda c: c[1])

        log_results: list[dict] = []

        # 3) Local optimization from each selected start
        for r in range(num_tries):
            x0 = samples[r][0]
            init_fun = samples[r][1]

            # SPSA coarse optimization
            res_spsa = SPSA(maxiter=max_iter_spa).minimize(fun=self.objf_avg, x0=x0)

            # COBYLA refinement
            res_cobyla = COBYLA(maxiter=max_iter_cobyla).minimize(fun=self.objf_avg, x0=res_spsa.x)
            
            # Evaluate the probability of sampling an optimal permutation
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

    
    def run_experiments(
        instance_pattern: str = "tai20_5_*.fsp",
        jobs_list: list[int] = [4, 5, 6],
        runs_per_instance: int = 3
    ) -> None:
        """
        Batch launcher over instances and job sizes; appends results to a CSV.

        Parameters
        ----------
        instance_pattern : str
            Glob pattern selecting the instance files to process.
        jobs_list : list[int]
            List of job counts (nj) to evaluate for each instance.
        runs_per_instance : int
            Number of repeated runs per (instance, nj) combination.

        Returns
        -------
        None
        """
        all_logs: list[dict] = []
        os.makedirs("results", exist_ok=True)

        for nj in jobs_list:
            for instance_file in sorted(glob.glob(instance_pattern)):
                instance_name = os.path.basename(instance_file)
                print(f"Processing instance: {instance_name}, Jobs: {nj}")

                # Load instance and set up VQE for the chosen number of jobs
                prob = PFSProblem.load(instance_file)
                vqe = VQE(prob, nj)
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
    VQE.run_experiments("../Taillard/tai20_5_*.fsp", jobs_list=[2, 3, 4, 5, 6], runs_per_instance=5)
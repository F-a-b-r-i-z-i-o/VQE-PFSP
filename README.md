# VQE for the Permutation Flow Shop Problem (PFSP)

This repository contains a minimal research scaffold to explore a **Variational Quantum Eigensolver–style** approach (with a hardware-efficient ansatz) for the **Permutation Flow Shop Problem (PFSP)**. It includes:

- A small PFSP utility with exact evaluation and brute-force optimum for tiny instances.
- A VQE runner that builds a `RealAmplitudes` circuit, samples bitstrings, decodes them into job permutations, and optimizes the **average makespan** using SPSA → COBYLA.

---

## Reference / Paper

If you use this code, please cite and see the companion paper:

- **A Variational Quantum Algorithm for the Permutation Flow Shop Scheduling Problem** — Baioletti, Fagiolo, Oddi, Rasconi (GECCO ’25 Companion, Malaga). [[PDF](vqe-pfsp.pdf)]

---

## Contents

- `problem.py` – PFSP utilities:
  - Load Taillard-like instances.
  - Map integers/bitstrings to permutations (Lehmer-like decoding).
  - Compute makespan for a permutation.
  - (For small `n`) brute-force the optimum.
- `vqe_pfsp.py` – VQE experiment driver:
  - Builds a `RealAmplitudes` ansatz sized to ~`log2(n!)` qubits and measures all qubits.
  - Objective = empirical average makespan over measurement outcomes.
  - Optimizers: coarse **SPSA**, then **COBYLA**.
  - Batch runner over multiple instances and job sizes; results are appended to `results/results_vqe_pfsp.csv`.

---

## Installation

> Python ≥ 3.9 is recommended.

Install all dependencies from the pinned list:
```bash
pip install -r requirements.txt
```

> If you upgrade packages, make sure the sampler/optimizer APIs used in the code still match your versions.

---

## Data format (Taillard-like `.fsp`)

The loader expects a text format like Taillard’s PFSP instances:
- line 1: comment/header (ignored)
- line 2: `nj nm` (numbers of jobs and machines)
- line 3: comment/header (ignored)
- then **`nm` lines**, each with **`nj` integers**, listing the processing-time **column** (machine-wise).

> Internally, times are stored as an array `ptime[j, m]` (job-major, machine index `m` in columns).

Example snippet (not an actual instance):
```
Taillard PFSP instance
5 3
processing times by machine
2 5 7 4 3
1 3 9 2 4
6 4 2 8 5
```

---

## Quick start

1) Put your Taillard instances somewhere (e.g., `../Taillard/`).  
2) Run the batch experiments (from the repo root or adjust paths accordingly):
```bash
python vqe_pfsp.py
```
The default main runs:
- pattern: `../Taillard/tai20_5_*.fsp`
- jobs list: `[2, 3, 4, 5, 6]`
- runs per (instance, n_jobs): `5`

Results are (appended) to:
```
results/results_vqe_pfsp.csv
```

To customize, open `vqe_pfsp.py` and change the default arguments of `VQE.run_experiments(...)` or call it manually from a notebook/script.

---

## How it works

### 1) PFSP evaluation
- `PFSProblem.evaluate(p)` computes the **makespan** of permutation `p` using the standard flow-shop recurrence (machine “north–west” max dependency).  
- `PFSProblem.find_optima()` brute-forces all permutations to find the **optimal makespan** `fmin` (feasible only for small `n`).

### 2) Encoding permutations
- We treat a **bitstring** as an integer and decode it into a permutation using a Lehmer-like mixed-radix method (`int2perm`).  
- In the VQE loop, the measured bitstring is **reversed** (`bs[::-1]`) before decoding—matching the intended endianness.

### 3) Ansatz & sampling
- Number of qubits: `ceil(log2(n!))` (rounded via a `0.99 + log2(fact)` trick).  
- Ansatz: `RealAmplitudes(n_qubits, reps=...)` + **measure all** qubits.  
- Sampler: `StatevectorSampler` with a default of **1024 shots**.

### 4) Objective
- For parameters `θ`, simulate counts `counts[bitstring]`.  
- Map each bitstring to a permutation and evaluate its makespan.  
- **Objective** = empirical average makespan over samples.

### 5) Optimization
- Multi-start: randomly sample many `θ` and keep the best `num_tries` seeds.  
- Run **SPSA** (coarse) → **COBYLA** (refinement).  
- After optimizing, re-sample to estimate the **probability of drawing an optimal permutation** (i.e., with makespan `fmin`).

---

## CLI / API notes

- `VQE.create_ansatz(reps=1)` – build circuit and measure all qubits.  
- `VQE.objf_avg(params)` – compute the average makespan objective.  
- `VQE.run(...)` – single experiment with restarts; returns per-try logs.  
- `VQE.run_experiments(...)` – batch launcher over instances and job sizes; writes/extends a CSV.  

## License

Choose and add a license (e.g., MIT) if you plan to share this publicly.

# ✨ VQE for the Permutation Flow Shop Problem (PFSP)

A compact research scaffold to explore a **Variational Quantum Eigensolver** approach for the **Permutation Flow Shop Problem**.  

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-research-blue">
  <img alt="python" src="https://img.shields.io/badge/Python-%E2%89%A5%203.9-3776AB">
  <img alt="license" src="https://img.shields.io/badge/License-MIT-green">
</p>


---

## 🧭 Overview

- 🧩 **PFSP utilities** with exact evaluation and a brute‑force optimum for tiny instances.  
- ⚛️ **VQE runner** that builds a `RealAmplitudes` circuit, samples bitstrings, decodes them into job permutations, and optimizes the **average makespan** with **SPSA → COBYLA**.

---

## 📚 Reference / Paper

If you use this code, please cite and see the companion paper:

- **A Variational Quantum Algorithm for the Permutation Flow Shop Scheduling Problem** — Baioletti, Fagiolo, Oddi, Rasconi (GECCO ’25 Companion, Malaga). [[📄 PDF](vqe-pfsp.pdf)]

---

## 📦 Contents

- `problem.py` — **PFSP utilities**
  - 📥 Load Taillard‑like instances
  - 🔢 Map integers/bitstrings to permutations (Lehmer‑like decoding)
  - 🧮 Compute makespan for a permutation
  - 🧪 (For small `n`) brute‑force the optimum
- `vqe_pfsp.py` — **VQE experiment driver**
  - 🧱 Builds a `RealAmplitudes` ansatz sized to ~`log2(n!)` qubits and measures all qubits
  - 🎯 Objective = empirical **average makespan** over measurement outcomes
  - 🛠️ Optimizers: coarse **SPSA**, then **COBYLA**
  - 📝 Batch runner over multiple instances and job sizes; results appended to `results/results_vqe_pfsp.csv`

---

## ⚙️ Installation

> [!NOTE]
> **Python ≥ 3.9** is recommended.

Install all dependencies from the pinned list:

```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> If you upgrade packages, make sure the sampler/optimizer APIs used in the code still match your versions.

---

## 📂 Data format (Taillard‑like `.fsp`)

The loader expects a text format like Taillard’s PFSP instances:

- line 1: comment/header (ignored)  
- line 2: `nj nm` (numbers of jobs and machines)  
- line 3: comment/header (ignored)  
- then **`nm` lines**, each with **`nj` integers**, listing the processing‑time **column** (machine‑wise).

> Internally, times are stored as an array `ptime[j, m]` (job‑major, machine index `m` in columns).

**Example** (not an actual instance):

```
Taillard PFSP instance
5 3
processing times by machine
2 5 7 4 3
1 3 9 2 4
6 4 2 8 5
```

---

## 🚀 Quick start

1. Put your Taillard instances somewhere (e.g., `../Taillard/`).  
2. Run the batch experiments (from the repo root or adjust paths accordingly):

```bash
python vqe_pfsp.py
```

**Default main runs**:

- pattern: `../Taillard/tai20_5_*.fsp`
- jobs list: `[2, 3, 4, 5, 6]`
- runs per (instance, n_jobs): `5`

**Results** are appended to:

```
results/results_vqe_pfsp.csv
```

To customize, open `vqe_pfsp.py` and change the default arguments of `VQE.run_experiments(...)` or call it manually from a notebook/script.

> [!TIP]
> For reproducibility, set a NumPy seed at the entry point before running experiments.

---

## ✅ At a glance

- 🧠 Encoding via Lehmer‑like decoding from measured bitstrings  
- 🔢 Qubits ≈ `(log2(n!))`  
- 📈 Objective: average makespan over shot samples  
- 🧪 Small‑`n` brute force provides a ground‑truth optimum for sanity checks

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ⚛️ &nbsp;and a pinch of 🧪</p>


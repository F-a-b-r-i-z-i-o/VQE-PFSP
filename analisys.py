import pandas as pd

file_path = "results_VQE_NO_WS_2-6/results_vqe_pfsp.csv"

df = pd.read_csv(file_path)

df = df[df["run"].between(1, 5)]

grouped = df.groupby(["instance", "n_job"])["prob_opt"].max().reset_index()

df_filtered = grouped[grouped["prob_opt"] > 0.8]

matrix = df_filtered.pivot(index="n_job", columns="instance", values="prob_opt")

matrix = matrix.fillna(0)

latex = matrix.to_latex(float_format="%.3f")

output_file = "matrice_prob_opt.tex"

with open(output_file, "w") as f:
    f.write(latex)

print(f"Matrice LaTeX salvata in: {output_file}")

import pandas as pd

file_path = "results_VQE_NO_WS_2-6/results_vqe_pfsp.csv"

df = pd.read_csv(file_path)

df = df[df["run"].between(1, 5)]

best_per_run = (
    df.groupby(["instance", "n_job", "run"])["prob_opt"]
      .max()
      .reset_index(name="best_prob_opt")
)

avg_prob = (
    best_per_run.groupby(["instance", "n_job"])["best_prob_opt"]
               .mean()
               .reset_index(name="prob_opt")
)

df_filtered = avg_prob[avg_prob["prob_opt"] > 0.8]

matrix = df_filtered.pivot(index="instance", columns="n_job", values="prob_opt")
matrix = matrix.fillna(0)


latex = matrix.to_latex(float_format="%.3f")

output_file = "matrice_prob_opt_mean.tex"
with open(output_file, "w") as f:
    f.write(latex)

print(f"Matrice LaTeX salvata in: {output_file}")
print("Ecco un'anteprima:")
print(latex)


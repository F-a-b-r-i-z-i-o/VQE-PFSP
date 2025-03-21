import pandas as pd


file_path = "results_VQE_NO_WS_2-6/results_vqe_pfsp.csv"
df = pd.read_csv(file_path)
df_filtered = df[df["prob_opt"] > 0.8]
matrix = df_filtered.pivot_table(index="instance", columns="n_job", aggfunc="size", fill_value=0)

latex = matrix.to_latex()

output_file = "matrice_prob_opt.tex"
with open(output_file, "w") as f:
    f.write(latex)

print(f"Matrice LaTeX salvata in: {output_file}")
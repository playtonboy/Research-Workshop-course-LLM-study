# Setup
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

def ensure_installed(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_installed("statsmodels")

# 1. loading data
df = pd.read_csv("results_final.csv")  

df["Correct_bin"] = (df["Correct"] == "Yes").astype(int)
df["Family"] = df["Family"].astype("category")
df["Model"] = df["Model"].astype("category")
df["Question_ID"] = df["Question_ID"].astype("category")
df["Run"] = df["Run"].astype("category")

# 2. models
size_map = {
    "llama-3.2-1b-instruct":                1,
    "llama-3.2-3b-instruct":                3,
    "meta-llama-3-8b-instruct":             8,
    "mistralai/ministral-3-3b":             3,
    "mistralai/mistral-7b-instruct-v0.3":   7,
    "ministral-3-14b-instruct-2512":        14,
    "qwen2.5-1.5b-instruct":                1.5,
    "deepseek-r1-distill-llama-8b":         8,
    "deepseek-r1-mourse-14b-instruct-v0.2": 14,
}

df["size_b"] = df["Model"].map(size_map)

# 3. model colors
family_colors = {
    "llama":    "mediumslateblue",
    "mistral":  "orange",
    "deepseek": "mediumseagreen",
}

# 4. descriptive statistics
design_summary = {
    "n_rows": len(df),
    "n_models": df["Model"].nunique(),
    "n_families": df["Family"].nunique(),
    "n_runs": df["Run"].nunique(),
    "n_questions": df["Question_ID"].nunique(),
}

# 5. run-level accuracy (per model × run)
run_acc = (
    df.groupby(["Model", "Run"])["Correct_bin"]
      .mean()
      .reset_index(name="run_accuracy")
)
run_acc_summary = (
    run_acc.groupby("Model")["run_accuracy"]
           .agg(mean_run_acc="mean", sd_run_acc="std")
           .reset_index()
           .sort_values("mean_run_acc", ascending=False)
)

# 6. accuracy per model (barplot)
run_acc_summary = run_acc_summary.merge(
    df[["Model", "Family"]].drop_duplicates(),
    on="Model",
    how="left"
)

colors = run_acc_summary["Family"].map(family_colors)
plt.figure(figsize=(10, 6))
plt.bar(run_acc_summary["Model"], run_acc_summary["mean_run_acc"], color=colors)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Mean accuracy per model")
plt.tight_layout()
plt.show()

# 7. accuracy vs. model size by family
acc_size_family = (
    df.groupby(["Family", "Model", "size_b"])["Correct_bin"]
      .mean()
      .reset_index(name="accuracy")
)

plt.figure(figsize=(8, 5))
for fam, sub in acc_size_family.groupby("Family"):
    sub = sub.sort_values("size_b")
    plt.plot(
        sub["size_b"],
        sub["accuracy"],
        marker="o",
        label=fam,
        color=family_colors.get(fam, "gray")
    )

plt.ylim(0, 1)
plt.xlabel("Model size (billions)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. model size by family")
plt.legend(title="Family")
plt.tight_layout()
plt.show()

# 8. accuracy vs. model size by family
acc_size_family = (
    df.groupby(["Family", "Model", "size_b"])["Correct_bin"]
      .mean()
      .reset_index(name="accuracy")
)
print(acc_size_family[["Family", "Model", "size_b", "accuracy"]].to_string())
acc_size_family = acc_size_family.dropna(subset=["size_b", "accuracy"])

families = sorted(acc_size_family["Family"].unique())
x_map = {fam: i for i, fam in enumerate(families)}

fig, ax = plt.subplots(figsize=(9, 6))

for fam, sub in acc_size_family.groupby("Family"):
    sub = sub.sort_values("size_b").reset_index(drop=True)
    n = len(sub)
    x_pos = np.full(len(sub), x_map[fam])

    sizes = 100 + sub["accuracy"].values ** 2 * 2000

    ax.scatter(
        x=x_pos,
        y=sub["size_b"].values,
        s=sizes,
        color=family_colors.get(fam, "gray"),
        alpha=0.80,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

    for x, y, acc in zip(x_pos, sub["size_b"].values, sub["accuracy"].values):
        ax.annotate(
            f"{acc*100:.0f}%",
            xy=(x, y),
            xytext=(x, y),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            zorder=4,
        )

ax.set_xticks(list(x_map.values()))
ax.set_xticklabels(list(x_map.keys()))
ax.set_xlabel("Model family")
ax.set_ylabel("Model size (billions of parameters)")
ax.set_title("Model size vs. family")
ax.set_ylim(0, 16)
ax.set_yticks(np.arange(0, 16 + 0.1, 2))
ax.grid(axis="y", alpha=0.2)
ax.margins(x=0.3, y=0.2)

plt.tight_layout()
plt.show()

# 9. question-level / item difficulty
acc_question = (
    df.groupby("Question_ID")["Correct_bin"]
      .mean()
      .reset_index(name="accuracy")
      .sort_values("Question_ID")
)
print("\nAccuracy per question:")
print(acc_question)

acc_model_question = (
    df.groupby(["Model", "Question_ID"])["Correct_bin"]
      .mean()
      .reset_index(name="accuracy")
)
heat = acc_model_question.pivot(index="Model", columns="Question_ID", values="accuracy")

plt.figure(figsize=(10, 6))
plt.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(label="Accuracy")
plt.xticks(ticks=np.arange(heat.shape[1]), labels=heat.columns, rotation=90)
plt.yticks(ticks=np.arange(heat.shape[0]), labels=heat.index)
plt.title("Accuracy heatmap by model and question")
plt.xlabel("Question ID")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# 10. Mixed-effects-style logistic regression (GLM with cluster-robust SE)
# Logistic regression: Correct ~ Family * size_b ; cluster standard errors by Question_ID to account for repeated questions
formula = "Correct_bin ~ C(Family) * size_b"
logit_model = smf.glm(formula=formula,
                      data=df,
                      family=sm.families.Binomial())
logit_result = logit_model.fit(cov_type="cluster",
                               cov_kwds={"groups": df["Question_ID"]})

print("\nLogistic regression with cluster-robust SE by Question_ID:")
print(logit_result.summary())

# 11. table for the report
params   = logit_result.params
se       = logit_result.bse
zvals    = logit_result.tvalues
pvals    = logit_result.pvalues
ci       = logit_result.conf_int()

label_map = {
    "Intercept":                       "Intercept",
    "C(Family)[T.llama]":              "Family: llama",
    "C(Family)[T.mistral]":            "Family: mistral",
    "size_b":                          "Model size (B)",
    "C(Family)[T.llama]:size_b":       "Family: llama × size",
    "C(Family)[T.mistral]:size_b":     "Family: mistral × size",
}

rows = []
for key in label_map:
    b    = params[key]
    s    = se[key]
    OR   = np.exp(b)
    z    = zvals[key]
    p    = pvals[key]
    lo   = np.exp(ci.loc[key, 0])
    hi   = np.exp(ci.loc[key, 1])
    ci_str = f"[{lo:.2f}, {hi:.2f}]"
    p_str  = f".{int(round(p * 1000)):03d}"   
    rows.append({
        "Predictor": label_map[key],
        "b":         f"{b:.2f}",
        "SE":        f"{s:.2f}",
        "OR":        f"{OR:.2f}" if p < .05 else f"{OR:.2f}",
        "95% CI":    ci_str,
        "z":         f"{z:.2f}",
        "p":         f"{p_str}" if p < .05 else p_str,
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False))
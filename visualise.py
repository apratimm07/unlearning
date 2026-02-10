import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

FILE_PATH = "results_tofu_olmo_0.pkl"

sns.set_theme(style="whitegrid", context="talk")

def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_matrices(identity, proposed):
    identity.to_csv("identity_influence_scores.csv")
    proposed.to_csv("datainf_influence_scores.csv")

    with open("identity_matrix.txt", "w") as f:
        f.write(identity.to_string())

    with open("proposed_matrix.txt", "w") as f:
        f.write(proposed.to_string())

def print_basic_stats(identity, proposed):
    print("\n===== BASIC INFO =====")
    print("Identity shape:", identity.shape)
    print("Proposed shape:", proposed.shape)

    print("\nDataInf Absolute Influence Stats")
    abs_vals = np.abs(proposed.values.flatten())
    print("Max:", abs_vals.max())
    print("Mean:", abs_vals.mean())
    print("Std:", abs_vals.std())

def heatmaps(identity, proposed):
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    subset_tr = min(100, identity.shape[0])
    subset_val = min(40, identity.shape[1])

    sns.heatmap(identity.iloc[:subset_tr, :subset_val], cmap="RdBu_r", center=0, ax=axes[0])
    axes[0].set_title("Identity Influence (Gradient Dot Product)")

    sns.heatmap(proposed.iloc[:subset_tr, :subset_val], cmap="RdBu_r", center=0, ax=axes[1])
    axes[1].set_title("DataInf Influence (Hessian Corrected)")

    plt.suptitle("Influence Heatmap Comparison")
    plt.savefig("viz_heatmaps.png", dpi=300)
    plt.close()

def distributions(identity, proposed):
    plt.figure(figsize=(12,6))
    sns.kdeplot(identity.values.flatten(), label="Identity", fill=True, alpha=0.4)
    sns.kdeplot(proposed.values.flatten(), label="DataInf", fill=True, alpha=0.4)
    plt.title("Distribution of Influence Scores")
    plt.xlabel("Influence Score")
    plt.legend()
    plt.savefig("viz_distributions.png", dpi=300)
    plt.close()

def top_k_samples(proposed, k=10):
    val_idx = 0
    scores = proposed[val_idx].sort_values()
    combined = pd.concat([scores.head(k), scores.tail(k)])

    colors = ["green" if x < 0 else "red" for x in combined]

    plt.figure(figsize=(12,7))
    combined.plot(kind="barh", color=colors)
    plt.title("Top Influential Training Samples (Forget Query 0)")
    plt.xlabel("Influence Score")
    plt.savefig("viz_topk.png", dpi=300)
    plt.close()

def concentration_curve(identity, proposed):
    plt.figure(figsize=(10,8))
    for name, df in [("Identity", identity), ("DataInf", proposed)]:
        abs_inf = np.sort(np.abs(df.values.flatten()))[::-1]
        cum_inf = np.cumsum(abs_inf) / np.sum(abs_inf)
        plt.plot(np.linspace(0,100,len(cum_inf)), cum_inf*100, label=name)

    plt.title("Influence Concentration Curve")
    plt.xlabel("Percentage of Sample Pairs")
    plt.ylabel("Percentage of Total Influence")
    plt.legend()
    plt.savefig("viz_concentration.png", dpi=300)
    plt.close()

def correlation_plot(identity, proposed):
    plt.figure(figsize=(10,10))
    plt.scatter(identity.values.flatten(), proposed.values.flatten(), alpha=0.2, s=2)
    plt.xlabel("Identity Score")
    plt.ylabel("DataInf Score")
    plt.title("Correlation Between Identity and DataInf")
    plt.savefig("viz_correlation.png", dpi=300)
    plt.close()

def variance_boxplot(proposed):
    plt.figure(figsize=(15,6))
    sns.boxplot(data=proposed.iloc[:, :15], palette="Set3")
    plt.title("Influence Variance Across Forget Queries")
    plt.ylabel("Influence Score")
    plt.savefig("viz_boxplot.png", dpi=300)
    plt.close()

def global_ranking(proposed):
    global_scores = proposed.abs().mean(axis=1).sort_values(ascending=False)[:20]
    plt.figure(figsize=(12,8))
    global_scores.plot(kind="barh")
    plt.title("Top Globally Influential Training Samples")
    plt.xlabel("Mean Absolute Influence")
    plt.savefig("viz_global_ranking.png", dpi=300)
    plt.close()

def main():
    if not os.path.exists(FILE_PATH):
        print("Results file not found")
        return

    data = load_data(FILE_PATH)
    identity = data["influence"]["identity"]
    proposed = data["influence"]["proposed"]

    save_matrices(identity, proposed)
    print_basic_stats(identity, proposed)

    print("\nGenerating visualizations...")

    heatmaps(identity, proposed)
    distributions(identity, proposed)
    top_k_samples(proposed)
    concentration_curve(identity, proposed)
    correlation_plot(identity, proposed)
    variance_boxplot(proposed)
    global_ranking(proposed)

    print("\nAll visualizations saved successfully.")

if __name__ == "__main__":
    main()

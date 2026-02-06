import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_all_visualizations(file_path):
    """
    Loads DataInf results and generates 6 detailed visual analyses.
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    inf_id = data['influence']['identity']   # Baseline
    inf_prop = data['influence']['proposed'] # DataInf
    
    sns.set_theme(style="whitegrid")
    print(f"Generating visualizations for {file_path}...")

    # --- 1. COMPARISON HEATMAPS ---
    # Shows the 'de-noising' effect of DataInf
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    subset_tr, subset_val = min(100, inf_id.shape[0]), min(40, inf_id.shape[1])
    
    sns.heatmap(inf_id.iloc[:subset_tr, :subset_val], ax=axes[0], cmap='RdBu_r', center=0)
    axes[0].set_title(r'Baseline: Identity ($\nabla L_{test} \cdot \nabla L_{train}$)', fontsize=15)
    
    sns.heatmap(inf_prop.iloc[:subset_tr, :subset_val], ax=axes[1], cmap='RdBu_r', center=0)
    axes[1].set_title(r'Proposed: DataInf ($\nabla L_{test}^\top (H+\lambda I)^{-1} \nabla L_{train}$)', fontsize=15)
    
    plt.suptitle("Heatmap: Training Sample Influence on Forget Set", fontsize=20)
    plt.savefig('viz_1_heatmaps.png', dpi=300)

    # --- 2. GLOBAL SCORE DISTRIBUTIONS ---
    # DataInf should have heavier tails (identifying stronger outliers)
    plt.figure(figsize=(12, 6))
    sns.kdeplot(inf_id.values.flatten(), label='Identity (Baseline)', fill=True, alpha=0.4)
    sns.kdeplot(inf_prop.values.flatten(), label='DataInf (Proposed)', fill=True, alpha=0.4)
    plt.title('Distribution of Influence Scores', fontsize=16)
    plt.xlabel('Influence Score')
    plt.legend()
    plt.savefig('viz_2_distributions.png', dpi=300)

    # --- 3. TOP-K INFLUENTIAL SAMPLES (BAR CHART) ---
    # Identify the specific 'culprits' for the first forget question
    val_idx = 0 
    top_k = 10
    scores = inf_prop[val_idx].sort_values()
    combined = pd.concat([scores.head(top_k), scores.tail(top_k)])
    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in combined]
    
    plt.figure(figsize=(12, 7))
    combined.plot(kind='barh', color=colors)
    plt.title(f'Top {top_k} Helpful/Harmful Samples for Forget Query {val_idx}', fontsize=16)
    plt.xlabel('Influence Score (Negative = Culprit/Helpful)')
    plt.savefig('viz_3_top_k_samples.png', dpi=300)

    # --- 4. CONCENTRATION CURVE (LORENZ-STYLE) ---
    # Shows if the signal is concentrated in a few training samples
    plt.figure(figsize=(10, 8))
    for name, df in [('Identity', inf_id), ('DataInf', inf_prop)]:
        abs_inf = np.sort(np.abs(df.values).flatten())[::-1]
        cum_inf = np.cumsum(abs_inf) / np.sum(abs_inf)
        plt.plot(np.linspace(0, 100, len(cum_inf)), cum_inf * 100, label=name)
        
    plt.title('Influence Concentration (Signal vs Noise)', fontsize=16)
    plt.xlabel('Percentage of Pairs (%)')
    plt.ylabel('Percentage of Total Influence (%)')
    plt.legend()
    plt.savefig('viz_4_concentration.png', dpi=300)

    # --- 5. METHOD CORRELATION ---
    # Visualizes how much the Hessian correction changes the ranking
    plt.figure(figsize=(10, 10))
    plt.scatter(inf_id.values.flatten(), inf_prop.values.flatten(), alpha=0.2, s=2)
    plt.title('Correlation: Identity vs. DataInf', fontsize=16)
    plt.xlabel('Identity Score')
    plt.ylabel('DataInf Score')
    plt.savefig('viz_5_correlation.png', dpi=300)

    # --- 6. QUERY VARIANCE BOXPLOT ---
    # Shows the stability of influence across different forget queries
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=inf_prop.iloc[:, :15], palette="Set3")
    plt.title('Influence Variance for First 15 Forget Queries', fontsize=16)
    plt.ylabel('Influence Score')
    plt.savefig('viz_6_boxplot.png', dpi=300)

    print("Success! All 6 charts saved in your current directory.")

if __name__ == "__main__":
    generate_all_visualizations("results_tofu_olmo_0.pkl")

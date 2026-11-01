import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    scenarios = [1, 2, 3, 4]
    
    # Store results
    base_min_sinr = []
    base_fairness = []
    drl_min_sinr = []
    drl_fairness = []
    
    scenario_names = [
        'S1: Same Spd/Dist',
        'S2: Vary Spd',
        'S3: Vary Dist',
        'S4: Vary Spd/Dist'
    ]
    
    file_map = {
        1: "S1_SameSpeed_SameDist",
        2: "S2_VarySpeed_SameDist",
        3: "S3_SameSpeed_VaryDist",
        4: "S4_VarySpeed_VaryDist"
    }

    print("Extracting Data...")
    
    for s in scenarios:
        # 1. Get Baseline Data
        base_file = f'../results/baseline/multi_baseline_s{s}_seed42.csv'
        if os.path.exists(base_file):
            base_df = pd.read_csv(base_file)
            means = [base_df[f'ue{i}_sinr'].mean() for i in range(5)]
            base_min_sinr.append(min(means))
            x = np.array(means)
            base_fairness.append((x.sum()**2) / (len(x) * (x**2).sum() + 1e-9))
        else:
            base_min_sinr.append(0)
            base_fairness.append(0)
            
        # 2. Get DRL Data (Average of the last 5 episodes of the 50-ep run)
        drl_file = f'../logs/training_{file_map[s]}.csv'
        if os.path.exists(drl_file) and os.path.getsize(drl_file) > 100:
            drl_df = pd.read_csv(drl_file)
            drl_min_sinr.append(drl_df['min_sinr'].tail(5).mean())
            drl_fairness.append(drl_df['fairness'].tail(5).mean())
        else:
            print(f"Warning: {drl_file} not ready yet. Using 0.")
            drl_min_sinr.append(0)
            drl_fairness.append(0)

    # === PLOTTING ===
    x_pos = np.arange(len(scenarios))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='#050d1a')
    fig.suptitle('Multi-UE (5 UEs) Performance: Baseline vs DRL (50-Ep Exploration)', 
                 color='white', fontsize=14, fontweight='bold')

    # Panel 1: Min SINR
    axes[0].set_facecolor('#0a1628')
    axes[0].bar(x_pos - width/2, base_min_sinr, width, label='Baseline (Ideal Sweep)', color='#ff4d6d')
    axes[0].bar(x_pos + width/2, drl_min_sinr, width, label='DRL (Exploration)', color='#00c8ff')
    axes[0].set_ylabel('Min SINR (Worst UE) [dB]', color='white')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(scenario_names, color='white')
    axes[0].legend()
    
    # Panel 2: Jain's Fairness
    axes[1].set_facecolor('#0a1628')
    axes[1].bar(x_pos - width/2, base_fairness, width, label='Baseline (Ideal Sweep)', color='#ff4d6d')
    axes[1].bar(x_pos + width/2, drl_fairness, width, label='DRL (Exploration)', color='#00e5a0')
    axes[1].set_ylabel("Jain's Fairness Index", color='white')
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(scenario_names, color='white')
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.15)
        ax.tick_params(colors='gray')
        for s in ax.spines.values(): s.set_color('#1a3a5c')

    plt.tight_layout()
    plt.savefig('../plots/multi_ue_evaluation.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to: plots/multi_ue_evaluation.png")

if __name__ == "__main__":
    main()

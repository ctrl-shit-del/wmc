import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_scenario_convergence(scenario_num, file_suffix, title_desc):
    file_path = f'../logs/training_{file_suffix}.csv'
    
    if not os.path.exists(file_path):
        print(f"Log file for Scenario {scenario_num} not found.")
        return

    df = pd.read_csv(file_path)
    
    # Calculate Moving Averages (window of 5 for our short 50-ep run, normally 20-30)
    window = min(5, len(df)) 
    df['reward_ma'] = df['reward'].rolling(window=window, min_periods=1).mean()
    df['fairness_ma'] = df['fairness'].rolling(window=window, min_periods=1).mean()
    df['loss_ma'] = df['loss'].rolling(window=window, min_periods=1).mean()

    # Apply Dark Theme
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), facecolor='#0b0f19')
    fig.patch.set_facecolor('#0b0f19')
    
    fig.suptitle(f'DQN Training Convergence — Scenario {scenario_num}: {title_desc}\n(50 Episodes: Exploration Phase)', 
                 color='white', fontsize=14, fontweight='bold')

    # Panel 1: Reward
    ax1 = axes[0]
    ax1.set_facecolor('#0b0f19')
    ax1.plot(df['episode'], df['reward'], color='#00c8ff', alpha=0.3, linewidth=1)
    ax1.plot(df['episode'], df['reward_ma'], color='#00c8ff', linewidth=2, label=f'Reward ({window}-ep MA)')
    ax1.set_ylabel('Reward', color='white')
    ax1.grid(color='#1a2436', linestyle='-', linewidth=0.5)
    ax1.legend(facecolor='#1c283d', edgecolor='none')

    # Panel 2: Jain's Fairness
    ax2 = axes[1]
    ax2.set_facecolor('#0b0f19')
    ax2.plot(df['episode'], df['fairness'], color='#ff9f43', alpha=0.3, linewidth=1)
    ax2.plot(df['episode'], df['fairness_ma'], color='#ff9f43', linewidth=2, label=f"Jain's Fairness ({window}-ep MA)")
    ax2.set_ylabel('Fairness Index', color='white')
    ax2.set_ylim(0, 1.1)
    ax2.grid(color='#1a2436', linestyle='-', linewidth=0.5)
    ax2.legend(facecolor='#1c283d', edgecolor='none')

    # Panel 3: Loss
    ax3 = axes[2]
    ax3.set_facecolor('#0b0f19')
    ax3.plot(df['episode'], df['loss'], color='#feca57', alpha=0.3, linewidth=1)
    ax3.plot(df['episode'], df['loss_ma'], color='#feca57', linewidth=2, label=f'Training Loss ({window}-ep MA)')
    ax3.set_xlabel('Episode', color='white')
    ax3.set_ylabel('Loss', color='white')
    ax3.grid(color='#1a2436', linestyle='-', linewidth=0.5)
    ax3.legend(facecolor='#1c283d', edgecolor='none')

    # Formatting tweaks
    for ax in axes:
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_color('#1c283d')

    plt.tight_layout()
    save_path = f'../plots/convergence_S{scenario_num}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot for Scenario {scenario_num} -> {save_path}")

def main():
    scenarios = {
        1: ("S1_SameSpeed_SameDist", "Same Spd/Dist"),
        2: ("S2_VarySpeed_SameDist", "Vary Spd/Same Dist"),
        3: ("S3_SameSpeed_VaryDist", "Same Spd/Vary Dist"),
        4: ("S4_VarySpeed_VaryDist", "Vary Spd/Dist")
    }
    
    print("Generating Multi-UE Convergence Plots...")
    for s_num, (s_file, s_desc) in scenarios.items():
        plot_scenario_convergence(s_num, s_file, s_desc)

if __name__ == "__main__":
    main()

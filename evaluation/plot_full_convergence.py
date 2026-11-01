import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_4_panel_convergence(scenario_num, s_name, title_desc):
    expl_file = f'../logs/training_{s_name}.csv'
    exploit_file = f'../logs/resume_{s_name}.csv'
    
    # 1. Load Exploration Data (50 episodes)
    if os.path.exists(expl_file):
        df_expl = pd.read_csv(expl_file)
    else:
        print(f"Warning: {expl_file} not found. Skipping S{scenario_num}.")
        return
        
    # 2. Load Exploitation Data (200 episodes)
    if os.path.exists(exploit_file):
        df_exploit = pd.read_csv(exploit_file)
        # Shift the episodes to continue sequentially (50 to 249)
        df_exploit['episode'] = df_exploit['episode'] + len(df_expl)
    else:
        print(f"Warning: {exploit_file} not found. Skipping S{scenario_num}.")
        return

    # 3. Stitch them together
    df = pd.concat([df_expl, df_exploit], ignore_index=True)
    
    # Handle column names (some logs use 'switches', some might use 'switch_rate')
    switch_col = 'switches' if 'switches' in df.columns else 'switch_rate'
    if switch_col not in df.columns: df[switch_col] = 0

    # 4. Calculate Moving Averages (Window=10 is perfect for 250 episodes)
    w = 10
    df['sinr_ma'] = df['min_sinr'].rolling(window=w, min_periods=1).mean()
    df['fair_ma'] = df['fairness'].rolling(window=w, min_periods=1).mean()
    df['switch_ma'] = df[switch_col].rolling(window=w, min_periods=1).mean()
    df['loss_ma'] = df['loss'].rolling(window=w, min_periods=1).mean()

    # === PLOTTING SETUP ===
    bg_color = '#0b0f19'
    grid_color = '#1a2436'
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), facecolor=bg_color, sharex=True)
    fig.patch.set_facecolor(bg_color)
    
    fig.suptitle(f'DQN Convergence — Scenario {scenario_num}: {title_desc}\n(250 Episodes: 50 Exploration + 200 Exploitation)', 
                 color='white', fontsize=15, fontweight='bold')

    # Panel 1: Min SINR
    ax1 = axes[0]
    ax1.set_facecolor(bg_color)
    ax1.plot(df['episode'], df['min_sinr'], color='#00c8ff', alpha=0.2, linewidth=1)
    ax1.plot(df['episode'], df['sinr_ma'], color='#00c8ff', linewidth=2.5, label='Min SINR (10-ep MA)')
    ax1.axvline(x=len(df_expl), color='#ffcc00', linestyle='--', linewidth=1.5, label='Exploitation Phase Begins')
    ax1.set_ylabel('Min SINR [dB]', color='white')
    ax1.legend(facecolor='#1c283d', edgecolor='none', labelcolor='white')

    # Panel 2: Jain's Fairness
    ax2 = axes[1]
    ax2.set_facecolor(bg_color)
    ax2.plot(df['episode'], df['fairness'], color='#00e5a0', alpha=0.2, linewidth=1)
    ax2.plot(df['episode'], df['fair_ma'], color='#00e5a0', linewidth=2.5, label="Jain's Fairness (10-ep MA)")
    ax2.axvline(x=len(df_expl), color='#ffcc00', linestyle='--', linewidth=1.5)
    ax2.set_ylabel('Fairness Index', color='white')
    ax2.set_ylim(0, 1.1)
    ax2.legend(facecolor='#1c283d', edgecolor='none', labelcolor='white')

    # Panel 3: Switch Rate
    ax3 = axes[2]
    ax3.set_facecolor(bg_color)
    ax3.plot(df['episode'], df[switch_col], color='#ff9f43', alpha=0.2, linewidth=1)
    ax3.plot(df['episode'], df['switch_ma'], color='#ff9f43', linewidth=2.5, label='Beam Switch Rate (10-ep MA)')
    ax3.axvline(x=len(df_expl), color='#ffcc00', linestyle='--', linewidth=1.5)
    ax3.set_ylabel('Switch Rate', color='white')
    ax3.legend(facecolor='#1c283d', edgecolor='none', labelcolor='white')

    # Panel 4: Loss
    ax4 = axes[3]
    ax4.set_facecolor(bg_color)
    ax4.plot(df['episode'], df['loss'], color='#feca57', alpha=0.2, linewidth=1)
    ax4.plot(df['episode'], df['loss_ma'], color='#feca57', linewidth=2.5, label='Training Loss (10-ep MA)')
    ax4.axvline(x=len(df_expl), color='#ffcc00', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Episode', color='white', fontsize=12)
    ax4.set_ylabel('Loss', color='white')
    
    # In Reinforcement learning, loss spikes are huge, so a log scale often looks better.
    # We will use linear here, but limit the Y-axis if the exploration loss blew up to 80,000
    max_exploit_loss = df_exploit['loss'].max()
    ax4.set_ylim(0, max_exploit_loss * 2) # Focuses the chart on the stable part
    ax4.legend(facecolor='#1c283d', edgecolor='none', labelcolor='white')

    # Global Formatting
    for ax in axes:
        ax.grid(color=grid_color, linestyle='-', linewidth=0.5)
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_color('#1c283d')

    plt.tight_layout()
    save_path = f'../plots/convergence_full_S{scenario_num}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved full convergence plot for S{scenario_num} -> {save_path}")

def main():
    scenarios = {
        1: ("S1_SameSpeed_SameDist", "Same Spd / Same Dist"),
        2: ("S2_VarySpeed_SameDist", "Vary Spd / Same Dist"),
        3: ("S3_SameSpeed_VaryDist", "Same Spd / Vary Dist"),
        4: ("S4_VarySpeed_VaryDist", "Vary Spd / Vary Dist")
    }
    
    print("Generating 250-Episode Convergence Plots...")
    os.makedirs('../plots', exist_ok=True)
    
    for s_num, (s_file, s_desc) in scenarios.items():
        plot_4_panel_convergence(s_num, s_file, s_desc)
        
    print("\nDONE! Check your ~/beam_project/plots/ folder.")

if __name__ == "__main__":
    main()

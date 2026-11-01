"""
evaluate_multiue.py
Evaluate trained DRL agents for all 4 scenarios.
Generates per-scenario and cross-scenario comparison plots.

Metrics:
- Min SINR (fairness — worst UE quality)
- Mean SINR (average across all UEs)
- Jain's Fairness Index
- Beam switch rate
- Per-UE SINR distribution

Baselines:
- Round-robin UE scheduling (no DRL)
- Nearest-UE greedy (always serve closest UE)
"""

import sys
import os
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.expanduser(
    '~/beam_project/agent'))

from dqn_multiue_agent import MultiUeAgent
from dqn_multiue_network import (
    build_state_vector, jains_fairness, NUM_UE
)
from train_multiue import (
    parse_multiue_state, make_pipes,
    STATE_PIPE, ACT_PIPE, SCENARIO_NAMES
)

STATE_PIPE = "/tmp/beam_state_multi"
ACT_PIPE   = "/tmp/beam_act_multi"
EVAL_SEEDS = [100, 101, 102, 103, 104]  # held-out seeds


# ── Run one evaluation episode ────────────────────────────
def run_eval_episode(agent_or_policy,
                      scenario: int,
                      seed: int,
                      sim_time: float = 10.0,
                      policy_type: str = "drl"
                      ) -> dict:
    """
    policy_type: 'drl', 'round_robin', 'greedy_nearest'
    """
    make_pipes()

    ns3_bin = os.path.expanduser("~/beam_project/ns3/ns3")
    ns3_dir = os.path.expanduser("~/beam_project/ns3")

    ns3_proc = subprocess.Popen(
        [ns3_bin, "run",
         (f"scratch/drl_multiue_sim "
          f"--seed={seed} "
          f"--simTime={sim_time} "
          f"--scenario={scenario} "
          f"--statePipe={STATE_PIPE} "
          f"--actPipe={ACT_PIPE}")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ns3_dir)

    sinr_per_ue  = [[] for _ in range(NUM_UE)]
    switches     = []
    fairnesses   = []
    prev_beam    = [-1] * NUM_UE
    rr_counter   = 0   # for round-robin

    try:
        state_f = open(STATE_PIPE, "r")
        act_f   = open(ACT_PIPE,   "w")
    except Exception:
        ns3_proc.terminate()
        ns3_proc.wait()
        return {}

    step = 0
    try:
        for line in state_f:
            ue_data = parse_multiue_state(line)
            if ue_data is None:
                continue

            sinr_list = [d['sinr_db'] for d in ue_data]

            # Select UE and beam based on policy
            if policy_type == "drl":
                state = build_state_vector(ue_data, step)
                ue_positions = [
                    (d['x_norm'], d['y_norm'])
                    for d in ue_data]
                ue, beam, _ = agent_or_policy.select_action(
                    state, ue_positions)

            elif policy_type == "round_robin":
                ue   = rr_counter % NUM_UE
                rr_counter += 1
                # Best beam for this UE (argmax Q or random)
                beam = np.random.randint(32)

            elif policy_type == "greedy_nearest":
                # Serve UE with highest SINR (greedy)
                ue   = int(np.argmax(sinr_list))
                beam = np.random.randint(32)

            act_f.write(f"{ue},{beam}\n")
            act_f.flush()

            # Track metrics
            for i, s in enumerate(sinr_list):
                sinr_per_ue[i].append(s)
            if prev_beam[ue] >= 0 and beam != prev_beam[ue]:
                switches.append(1)
            fairnesses.append(jains_fairness(sinr_list))
            prev_beam[ue] = beam
            step += 1

    except Exception as e:
        if "Broken pipe" not in str(e):
            print(f"Error: {e}")
    finally:
        try:
            state_f.close()
            act_f.close()
        except Exception:
            pass
        ns3_proc.terminate()
        ns3_proc.wait()

    if step == 0:
        return {}

    return {
        "min_sinr":    np.mean([min(s) for s in
                                zip(*sinr_per_ue)]),
        "mean_sinr":   np.mean([np.mean(s)
                                for s in sinr_per_ue]),
        "max_sinr":    np.mean([max(s) for s in
                                zip(*sinr_per_ue)]),
        "fairness":    np.mean(fairnesses),
        "switch_rate": len(switches) / max(step, 1),
        "per_ue_sinr": [np.mean(s) for s in sinr_per_ue],
        "steps":       step,
    }


# ── Evaluate all scenarios ────────────────────────────────
def evaluate_all(cfg_path: str = "../config.json",
                  sim_time: float = 10.0):

    results = {}

    for sc in [1, 2, 3, 4]:
        sname = SCENARIO_NAMES[sc]
        print(f"\n{'='*55}")
        print(f"Evaluating Scenario {sc}: {sname}")
        print(f"{'='*55}")

        # Load trained DRL agent
        model_path = Path(
            f"../checkpoints/multiue_s{sc}_final.pt")
        if not model_path.exists():
            print(f"  Model not found: {model_path}")
            print(f"  Skipping scenario {sc}")
            continue

        agent = MultiUeAgent(cfg_path, sc)
        agent.load(str(model_path))
        agent.epsilon = 0.0  # pure exploitation
        print(f"  Loaded: {model_path}")

        sc_results = {
            "drl": [], "round_robin": [], "greedy": []
        }

        for seed in EVAL_SEEDS:
            print(f"  Seed {seed}:", end=" ")

            r_drl = run_eval_episode(
                agent, sc, seed, sim_time, "drl")
            r_rr  = run_eval_episode(
                None, sc, seed, sim_time, "round_robin")
            r_gr  = run_eval_episode(
                None, sc, seed, sim_time, "greedy_nearest")

            sc_results["drl"].append(r_drl)
            sc_results["round_robin"].append(r_rr)
            sc_results["greedy"].append(r_gr)

            print(f"DRL min_SINR={r_drl.get('min_sinr',0):.1f}dB "
                  f"fair={r_drl.get('fairness',0):.3f}")

        results[sc] = sc_results

    return results


# ── Plot generation ───────────────────────────────────────
def generate_plots(results: dict,
                    log_dir: str = "../logs"):

    plt.style.use('dark_background')
    COLORS = {
        "drl":         "#00c8ff",
        "round_robin": "#ff8c42",
        "greedy":      "#ff4d6d",
    }
    LABELS = {
        "drl":         "DRL (DQN)",
        "round_robin": "Round Robin",
        "greedy":      "Greedy (Max SINR)",
    }
    SCENARIOS = {
        1: "S1: Same Speed\nSame Distance",
        2: "S2: Vary Speed\nSame Distance",
        3: "S3: Same Speed\nVary Distance",
        4: "S4: Vary Speed\nVary Distance",
    }

    # ── Plot 1: 4-scenario min-SINR comparison ──────────
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), facecolor='#050d1a')
    fig.suptitle(
        "Multi-UE DRL Beam Management — 5 UEs, 28 GHz mmWave\n"
        "Min-SINR (Fairness Metric) Across 4 Scenarios",
        color='white', fontsize=13, fontweight='bold')

    for idx, sc in enumerate([1, 2, 3, 4]):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor('#0a1628')

        if sc not in results:
            ax.text(0.5, 0.5, "No data",
                    transform=ax.transAxes,
                    ha='center', color='gray')
            continue

        sc_res = results[sc]
        x_pos  = np.arange(3)
        bar_labels = ["DRL\n(DQN)",
                       "Round\nRobin",
                       "Greedy\n(Max SINR)"]
        bar_colors = [COLORS["drl"],
                       COLORS["round_robin"],
                       COLORS["greedy"]]
        keys = ["drl", "round_robin", "greedy"]

        means = []
        cis   = []
        for k in keys:
            vals = [r.get("min_sinr", 0)
                    for r in sc_res[k] if r]
            if vals:
                m = np.mean(vals)
                c = (stats.t.ppf(0.975, len(vals)-1)
                     * np.std(vals, ddof=1)
                     / np.sqrt(len(vals)))
                means.append(m)
                cis.append(c)
            else:
                means.append(0)
                cis.append(0)

        bars = ax.bar(x_pos, means, yerr=cis,
                       color=bar_colors, alpha=0.85,
                       capsize=6, width=0.5,
                       error_kw={'color':'white','lw':1.5})

        ax.set_title(SCENARIOS[sc],
                     color='#00c8ff', fontsize=11,
                     fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels,
                            color='#c0c0c0', fontsize=9)
        ax.set_ylabel("Min SINR (dB)", color='white')
        ax.tick_params(colors='gray')
        ax.grid(alpha=0.15, axis='y')
        for s in ax.spines.values():
            s.set_color('#1a3a5c')

    plt.tight_layout()
    out = f"{log_dir}/../plots/multiue_minsinr.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

    # ── Plot 2: Jain's Fairness across scenarios ────────
    fig, ax = plt.subplots(figsize=(12, 6),
                            facecolor='#050d1a')
    ax.set_facecolor('#0a1628')
    fig.suptitle(
        "Jain's Fairness Index — 5 UEs, All Scenarios",
        color='white', fontsize=13, fontweight='bold')

    sc_labels = [SCENARIOS[sc].replace('\n', ' ')
                 for sc in [1,2,3,4]
                 if sc in results]
    x    = np.arange(len(sc_labels))
    w    = 0.25
    keys = ["drl", "round_robin", "greedy"]

    for ki, k in enumerate(keys):
        means = []
        cis   = []
        for sc in [1,2,3,4]:
            if sc not in results:
                continue
            vals = [r.get("fairness", 0)
                    for r in results[sc][k] if r]
            if vals:
                m = np.mean(vals)
                c = (stats.t.ppf(0.975, len(vals)-1)
                     * np.std(vals, ddof=1)
                     / np.sqrt(len(vals))
                     if len(vals) > 1 else 0)
            else:
                m, c = 0, 0
            means.append(m)
            cis.append(c)

        ax.bar(x + ki * w, means, w,
               yerr=cis,
               label=LABELS[k],
               color=COLORS[k], alpha=0.85,
               capsize=5,
               error_kw={'color':'white','lw':1.2})

    ax.axhline(y=1.0, color='#00e5a0', ls='--',
               lw=1.5, alpha=0.6,
               label='Perfect fairness (1.0)')
    ax.set_xticks(x + w)
    ax.set_xticklabels(sc_labels,
                        color='white', fontsize=9)
    ax.set_ylabel("Jain's Fairness Index",
                   color='white')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.tick_params(colors='gray')
    ax.grid(alpha=0.15, axis='y')
    for s in ax.spines.values():
        s.set_color('#1a3a5c')

    plt.tight_layout()
    out = f"{log_dir}/../plots/multiue_fairness.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

    # ── Plot 3: Per-UE SINR for Scenario 4 (hardest) ───
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), facecolor='#050d1a')
    fig.suptitle(
        "Per-UE SINR Distribution — Scenario 4\n"
        "(Vary Speed + Vary Distance: Hardest Case)",
        color='white', fontsize=12, fontweight='bold')

    sc = 4
    if sc in results:
        titles = ["DRL (DQN)",
                   "Round Robin", "Greedy (Max SINR)"]
        keys   = ["drl", "round_robin", "greedy"]
        ue_labels = [
            "UE0\n40m 1.5m/s",  "UE1\n60m 8.3m/s",
            "UE2\n100m 16.7m/s", "UE3\n150m 27.8m/s",
            "UE4\n185m 38.9m/s"
        ]

        for ai, (k, title) in enumerate(
                zip(keys, titles)):
            ax = axes[ai]
            ax.set_facecolor('#0a1628')

            per_ue_means = []
            for ue_i in range(NUM_UE):
                vals = [r["per_ue_sinr"][ue_i]
                        for r in results[sc][k]
                        if r and "per_ue_sinr" in r]
                per_ue_means.append(
                    np.mean(vals) if vals else 0)

            colors = plt.cm.plasma(
                np.linspace(0.2, 0.9, NUM_UE))
            bars = ax.bar(range(NUM_UE),
                           per_ue_means,
                           color=colors, alpha=0.85)

            ax.set_title(title, color='#00c8ff',
                         fontsize=11, fontweight='bold')
            ax.set_xticks(range(NUM_UE))
            ax.set_xticklabels(ue_labels,
                                color='#c0c0c0',
                                fontsize=7.5)
            ax.set_ylabel("Mean SINR (dB)",
                           color='white')
            ax.tick_params(colors='gray')
            ax.grid(alpha=0.15, axis='y')
            for s in ax.spines.values():
                s.set_color('#1a3a5c')

    plt.tight_layout()
    out = f"{log_dir}/../plots/multiue_perue_sinr.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

    # ── Plot 4: Training convergence all 4 scenarios ────
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9), facecolor='#050d1a')
    fig.suptitle(
        "DRL Training Convergence — All 4 Scenarios\n"
        "(Min-SINR and Jain's Fairness per Episode)",
        color='white', fontsize=12, fontweight='bold')

    for idx, sc in enumerate([1,2,3,4]):
        ax = axes[idx//2][idx%2]
        ax.set_facecolor('#0a1628')
        log_f = Path(
            f"../logs/training_{SCENARIO_NAMES[sc]}.csv")
        if not log_f.exists():
            ax.text(0.5, 0.5, "No training log",
                    transform=ax.transAxes,
                    ha='center', color='gray')
            continue

        log = pd.read_csv(log_f)

        def smooth(x, w=20):
            return pd.Series(x).rolling(
                w, min_periods=1).mean()

        ax2 = ax.twinx()

        ax.plot(log['episode'],
                smooth(log['min_sinr']),
                color='#00c8ff', lw=2,
                label='Min SINR (dB)')
        ax2.plot(log['episode'],
                 smooth(log['fairness']),
                 color='#00e5a0', lw=1.5,
                 ls='--', label="Fairness")

        ax.set_title(SCENARIOS[sc],
                     color='#00c8ff', fontsize=10,
                     fontweight='bold')
        ax.set_xlabel("Episode", color='gray')
        ax.set_ylabel("Min SINR (dB)", color='#00c8ff')
        ax2.set_ylabel("Fairness", color='#00e5a0')
        ax.tick_params(colors='gray')
        ax2.tick_params(colors='gray')
        ax.grid(alpha=0.15)
        for s in ax.spines.values():
            s.set_color('#1a3a5c')

    plt.tight_layout()
    out = f"{log_dir}/../plots/multiue_convergence.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ── Summary table ─────────────────────────────────────────
def print_summary_table(results: dict):
    print("\n" + "="*75)
    print("MULTI-UE RESULTS SUMMARY — 5 UEs, 28 GHz mmWave")
    print("="*75)
    header = (f"{'Scenario':<35} {'Policy':<15} "
              f"{'MinSINR':>8} {'MeanSINR':>9} "
              f"{'Fairness':>9} {'SwitchRate':>11}")
    print(header)
    print("-"*75)

    for sc in [1, 2, 3, 4]:
        if sc not in results:
            continue
        sname = SCENARIO_NAMES[sc]
        keys  = ["drl", "round_robin", "greedy"]
        plabs = ["DRL (DQN)",
                  "Round Robin", "Greedy (MaxSINR)"]

        for k, pl in zip(keys, plabs):
            vals = [r for r in results[sc][k] if r]
            if not vals:
                continue
            min_s = np.mean([v["min_sinr"] for v in vals])
            mea_s = np.mean([v["mean_sinr"] for v in vals])
            fair  = np.mean([v["fairness"] for v in vals])
            sw    = np.mean([v["switch_rate"] for v in vals])
            print(f"{sname:<35} {pl:<15} "
                  f"{min_s:>7.2f}  {mea_s:>8.2f}  "
                  f"{fair:>8.4f}  {sw:>10.4f}")
        print("-"*75)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",
        default="../config.json")
    p.add_argument("--simtime",
        type=float, default=10.0)
    p.add_argument("--plots_only",
        action="store_true",
        help="Skip evaluation, just regenerate plots")
    args = p.parse_args()

    if args.plots_only:
        # Load saved logs and regenerate plots
        print("Regenerating plots from training logs...")
        generate_plots({}, "../logs")
    else:
        results = evaluate_all(args.cfg, args.simtime)
        print_summary_table(results)

        # Save raw results
        import json
        out = {str(sc): {
            k: [r for r in v if r]
            for k, v in sc_res.items()
        } for sc, sc_res in results.items()}

        Path("../results").mkdir(exist_ok=True)
        with open("../results/multiue_results.json","w") as f:
            json.dump(out, f, indent=2, default=float)
        print("\nSaved: results/multiue_results.json")

        generate_plots(results, "../logs")

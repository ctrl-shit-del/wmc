import sys
import os
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.expanduser(
    '~/beam_project/agent'))
from dqn_agent import DQNAgent

STATE_PIPE = "/tmp/beam_state"
ACT_PIPE   = "/tmp/beam_act"


def make_pipes():
    for p in [STATE_PIPE, ACT_PIPE]:
        if os.path.exists(p):
            os.remove(p)
        os.mkfifo(p)


def run_drl_episode(agent, seed, sim_time=10):
    make_pipes()

    ns3_bin = os.path.expanduser(
        "~/beam_project/ns3/ns3")
    ns3_dir = os.path.expanduser(
        "~/beam_project/ns3")

    ns3_proc = subprocess.Popen(
        [ns3_bin, "run",
         (f"scratch/drl_beam_sim "
          f"--seed={seed} "
          f"--simTime={sim_time} "
          f"--speed=1.5 "
          f"--statePipe={STATE_PIPE} "
          f"--actPipe={ACT_PIPE}")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ns3_dir)

    sinrs    = []
    switches = []
    prev_beam = -1

    try:
        state_f = open(STATE_PIPE, 'r')
        act_f   = open(ACT_PIPE,   'w')

        for line in state_f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue

            sinr_db = float(parts[0])
            ue_x_n  = float(parts[1])
            ue_y_n  = float(parts[2])

            state = np.zeros(16, dtype=np.float32)
            state[0] = sinr_db / 40.0
            state[2] = ue_x_n
            state[3] = ue_y_n

            # Frozen policy — no training
            action, _ = agent.select_action(
                state,
                ue_x_n * 200,
                ue_y_n * 200)

            act_f.write(f"{action}\n")
            act_f.flush()

            sinrs.append(sinr_db)
            if action != prev_beam and prev_beam >= 0:
                switches.append(1)
            prev_beam = action

    except Exception as e:
        if "Broken pipe" not in str(e):
            print(f"Seed {seed} error: {e}")
    finally:
        try:
            state_f.close()
            act_f.close()
        except Exception:
            pass
        ns3_proc.terminate()
        ns3_proc.wait()

    return {
        'sinr_mean':   np.mean(sinrs) if sinrs else 0,
        'sinr_std':    np.std(sinrs)  if sinrs else 0,
        'switch_rate': len(switches) / max(len(sinrs), 1),
        'n_steps':     len(sinrs),
    }


def main():
    cfg_path = "../config.json"
    model    = "../checkpoints/dqn_final.pt"
    seeds    = [1, 2, 3, 4, 5]
    sim_time = 10

    # Load frozen DRL agent
    agent = DQNAgent(cfg_path)
    agent.load(model)
    agent.epsilon = 0.0  # Pure exploitation
    print(f"Loaded: {model}")
    print(f"Epsilon set to 0.0 (frozen policy)")

    # ── DRL evaluation ────────────────────────────
    print("\nEvaluating DRL policy...")
    drl_results = []
    for seed in seeds:
        r = run_drl_episode(agent, seed, sim_time)
        drl_results.append(r)
        print(f"  Seed {seed}: SINR={r['sinr_mean']:.2f}dB "
              f"switches={r['switch_rate']:.3f} "
              f"steps={r['n_steps']}")

    # ── Load baseline results ─────────────────────
    print("\nLoading baseline results...")
    baseline_results = []
    base_dir = Path("../results/baseline")
    for seed in seeds:
        f = base_dir / f"baseline_seed{seed}.csv"
        if not f.exists():
            print(f"  Missing: {f}")
            continue
        df = pd.read_csv(f)
        baseline_results.append({
            'sinr_mean': df['sinr_db'].mean(),
            'sinr_std':  df['sinr_db'].std(),
            'switch_rate': 0.0,  # sweep has no switching
            'n_steps':   len(df),
        })
        print(f"  Seed {seed}: SINR="
              f"{baseline_results[-1]['sinr_mean']:.2f}dB "
              f"steps={baseline_results[-1]['n_steps']}")

    # ── Statistical comparison ────────────────────
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)

    base_sinrs = [r['sinr_mean'] for r in baseline_results]
    drl_sinrs  = [r['sinr_mean'] for r in drl_results]

    bm  = np.mean(base_sinrs)
    dm  = np.mean(drl_sinrs)
    bci = stats.t.ppf(0.975, len(base_sinrs)-1) * \
          np.std(base_sinrs, ddof=1) / \
          np.sqrt(len(base_sinrs))
    dci = stats.t.ppf(0.975, len(drl_sinrs)-1) * \
          np.std(drl_sinrs, ddof=1) / \
          np.sqrt(len(drl_sinrs))

    t_stat, p_val = stats.ttest_ind(
        drl_sinrs, base_sinrs, equal_var=False)
    gain = (dm - bm) / abs(bm) * 100

    print(f"Baseline SINR: {bm:.2f} ± {bci:.2f} dB")
    print(f"DRL SINR:      {dm:.2f} ± {dci:.2f} dB")
    print(f"Gain:          {gain:+.1f}%")
    print(f"Welch t-test:  t={t_stat:.3f}, p={p_val:.4f}")
    print(f"Significant:   "
          f"{'YES' if p_val < 0.05 else 'NO'}")

    # Save results
    Path("../results/drl").mkdir(
        parents=True, exist_ok=True)
    summary = {
        'baseline_sinr_mean': bm,
        'baseline_sinr_ci':   bci,
        'drl_sinr_mean':      dm,
        'drl_sinr_ci':        dci,
        'gain_pct':           gain,
        't_stat':             t_stat,
        'p_value':            p_val,
        'significant': bool(p_val < 0.05),
    }
    import json
    Path("../results/final_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("\nSaved: results/final_summary.json")


if __name__ == "__main__":
    main()

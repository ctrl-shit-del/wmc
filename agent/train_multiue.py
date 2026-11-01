import sys
import os
import json
import time
import subprocess
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser(
    "~/beam_project/agent"))

from dqn_multiue_agent import MultiUeAgent
from dqn_multiue_network import (
    build_state_vector, jains_fairness, NUM_UE
)


SCENARIO_NAMES = {
    1: "S1_SameSpeed_SameDist",
    2: "S2_VarySpeed_SameDist",
    3: "S3_SameSpeed_VaryDist",
    4: "S4_VarySpeed_VaryDist",
}


def make_pipes(state_pipe, act_pipe):
    for p in [state_pipe, act_pipe]:
        if os.path.exists(p):
            os.remove(p)
        os.mkfifo(p)


def parse_multiue_state(line, num_ue=NUM_UE):
    parts = line.strip().split(",")
    if len(parts) < num_ue * 6:
        return None
    ue_data = []
    for i in range(num_ue):
        o = i * 6
        ue_data.append({
            "sinr_db":   float(parts[o + 0]),
            "x_norm":    float(parts[o + 1]),
            "y_norm":    float(parts[o + 2]),
            "vx_norm":   float(parts[o + 3]),
            "vy_norm":   float(parts[o + 4]),
            "prev_beam": float(parts[o + 5]),
            "sinr_h1":   float(parts[o + 0]),
        })
    return ue_data


def train_scenario(scenario, cfg_path,
                   seed, n_episodes, sim_time):

    sname    = SCENARIO_NAMES[scenario]
    ckpt_dir = Path("../checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    log_dir  = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"training_{sname}.csv"

    agent = MultiUeAgent(cfg_path, scenario)
    state_pipe = f"/tmp/beam_state_multi_s{scenario}"
    act_pipe = f"/tmp/beam_act_multi_s{scenario}"

    ns3_bin = os.path.expanduser(
        "~/beam_project/ns3/ns3")
    ns3_dir = os.path.expanduser(
        "~/beam_project/ns3")

    log_f = log_path.open("w")
    log_f.write(
        "episode,reward,min_sinr,mean_sinr,"
        "max_sinr,fairness,switches,"
        "loss,epsilon\n")
    log_f.flush()

    print("\n" + "="*55)
    print(f"Scenario {scenario}: {sname}")
    print(f"Episodes:{n_episodes} SimTime:{sim_time}s")
    print("="*55)

    pbar = tqdm(range(n_episodes),
                desc=f"S{scenario}",
                unit="ep", ncols=85)

    for ep in pbar:
        make_pipes(state_pipe, act_pipe)

        ns3_proc = subprocess.Popen(
            [ns3_bin, "run",
             (f"scratch/drl_multiue_sim "
              f"--seed={seed + ep} "
              f"--simTime={sim_time} "
              f"--scenario={scenario} "
              f"--statePipe={state_pipe} "
              f"--actPipe={act_pipe}")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ns3_dir)

        try:
            state_f = open(state_pipe, "r")
            act_f   = open(act_pipe,   "w")
        except Exception as e:
            print(f"\nPipe error ep={ep}: {e}")
            ns3_proc.terminate()
            ns3_proc.wait()
            continue

        ep_rewards = []
        sinr_mins  = []
        sinr_means = []
        sinr_maxs  = []
        fairnesses = []
        switches   = []
        losses     = []
        prev_state = None
        prev_ue    = -1
        prev_beam  = [-1] * NUM_UE
        step       = 0

        try:
            for line in state_f:
                ue_data = parse_multiue_state(line)
                if ue_data is None:
                    continue

                # Build 48D state vector
                state = build_state_vector(
                    ue_data, step, 2800)

                ue_positions = [
                    (d["x_norm"], d["y_norm"])
                    for d in ue_data]

                ue, beam, inf_ms = agent.select_action(
                    state, ue_positions)

                act_f.write(f"{ue},{beam}\n")
                act_f.flush()

                sinr_list = [d["sinr_db"]
                             for d in ue_data]
                beam_changed = (
                    beam != prev_beam[ue]
                    and prev_beam[ue] >= 0)
                reward = agent.compute_reward(
                    sinr_list, ue, beam_changed)

                if prev_state is not None:
                    agent.buffer.push(
                        prev_state,
                        prev_ue,
                        prev_beam[prev_ue]
                            if prev_ue >= 0 else 0,
                        reward,
                        state,
                        False)

                loss = agent.train_step()

                ep_rewards.append(reward)
                sinr_mins.append(min(sinr_list))
                sinr_means.append(
                    float(np.mean(sinr_list)))
                sinr_maxs.append(max(sinr_list))
                fairnesses.append(
                    jains_fairness(sinr_list))
                if beam_changed:
                    switches.append(1)
                if loss > 0:
                    losses.append(loss)

                prev_state    = state.copy()
                prev_ue       = ue
                prev_beam[ue] = beam
                step         += 1

        except Exception as e:
            if "Broken pipe" not in str(e):
                print(f"\nEp {ep} error: {e}")
        finally:
            try:
                state_f.close()
                act_f.close()
            except Exception:
                pass
            ns3_proc.terminate()
            ns3_proc.wait()

        agent.episode = ep

        ep_r   = float(np.sum(ep_rewards)) if ep_rewards else 0.0
        min_s  = float(np.mean(sinr_mins))  if sinr_mins  else 0.0
        mean_s = float(np.mean(sinr_means)) if sinr_means else 0.0
        max_s  = float(np.mean(sinr_maxs))  if sinr_maxs  else 0.0
        fair   = float(np.mean(fairnesses)) if fairnesses else 0.0
        sw_r   = len(switches) / max(len(ep_rewards), 1)
        loss_m = float(np.mean(losses))     if losses     else 0.0

        log_line = (f"{ep},{ep_r:.3f},{min_s:.2f},"
                    f"{mean_s:.2f},{max_s:.2f},"
                    f"{fair:.4f},{sw_r:.3f},"
                    f"{loss_m:.6f},{agent.epsilon:.4f}\n")
        log_f.write(log_line)
        log_f.flush()

        pbar.set_postfix(
            R=f"{ep_r:.0f}",
            MinS=f"{min_s:.1f}",
            Fair=f"{fair:.3f}",
            eps=f"{agent.epsilon:.3f}",
            steps=step)

        if ep % 100 == 0 and ep > 0:
            agent.save(str(ckpt_dir /
                f"multiue_s{scenario}_ep{ep:04d}.pt"))

    agent.save(str(
        ckpt_dir / f"multiue_s{scenario}_final.pt"))
    log_f.close()
    print(f"\nScenario {scenario} done. "
          f"Log: {log_path}")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",
        type=str, default="1")
    parser.add_argument("--cfg",
        default="../config.json")
    parser.add_argument("--seed",
        type=int, default=42)
    parser.add_argument("--episodes",
        type=int, default=300)
    parser.add_argument("--simtime",
        type=float, default=3.0)
    args = parser.parse_args()

    if args.scenario == "all":
        scenarios = [1, 2, 3, 4]
    else:
        scenarios = [int(args.scenario)]

    for sc in scenarios:
        train_scenario(
            scenario   = sc,
            cfg_path   = args.cfg,
            seed       = args.seed,
            n_episodes = args.episodes,
            sim_time   = args.simtime)

    print("\n=== All scenarios complete ===")


if __name__ == "__main__":
    main()

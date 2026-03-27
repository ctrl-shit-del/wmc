import sys
import os
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm

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


def main():
    cfg_path   = "../config.json"
    checkpoint = "../checkpoints/dqn_final.pt"
    seed       = 1
    start_ep   = 500
    total_ep   = 2000

    cfg      = json.loads(
        Path(cfg_path).read_text())
    ckpt_dir = Path("../checkpoints")
    log_path = Path("../logs/training_log.csv")

    agent = DQNAgent(cfg_path)
    agent.load(checkpoint)
    agent.episode = start_ep
    agent.epsilon = 0.8
    print(f"Loaded checkpoint. epsilon=0.8")

    ns3_bin = os.path.expanduser(
        "~/beam_project/ns3/ns3")
    ns3_dir = os.path.expanduser(
        "~/beam_project/ns3")

    log_f = log_path.open("a")

    pbar = tqdm(range(start_ep, total_ep),
                desc="Resuming",
                unit="ep", ncols=80)

    for ep in pbar:
        make_pipes()

        ns3_proc = subprocess.Popen(
            [ns3_bin, "run",
             (f"scratch/drl_beam_sim "
              f"--seed={seed + ep} "
              f"--simTime=3 "
              f"--speed=1.5 "
              f"--statePipe={STATE_PIPE} "
              f"--actPipe={ACT_PIPE}")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ns3_dir)

        try:
            state_f = open(STATE_PIPE, "r")
            act_f   = open(ACT_PIPE,   "w")
        except Exception as e:
            ns3_proc.terminate()
            ns3_proc.wait()
            continue

        ep_rewards, sinrs = [], []
        switches,   losses = [], []
        prev_state = None
        prev_beam  = -1
        step       = 0

        try:
            for line in state_f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue

                sinr_db = float(parts[0])
                ue_x_n  = float(parts[1])
                ue_y_n  = float(parts[2])
                ue_vx   = float(parts[3])
                ue_vy   = float(parts[4])
                pb      = float(parts[5])

                state = np.array([
                    sinr_db / 40.0, 0.0,
                    ue_x_n, ue_y_n,
                    ue_vx,  ue_vy,
                    pb / 32.0,
                    sinr_db / 40.0,
                    sinr_db / 40.0,
                    sinr_db / 40.0,
                    sinr_db / 40.0,
                    sinr_db / 40.0,
                    pb / 32.0, pb / 32.0,
                    float(ep) / total_ep,
                    float(step) / 2800.0,
                ], dtype=np.float32)

                action, _ = agent.select_action(
                    state,
                    ue_x_n * 200,
                    ue_y_n * 200)

                act_f.write(f"{action}\n")
                act_f.flush()

                beam_changed = (
                    action != prev_beam
                    and prev_beam >= 0)
                reward = agent.compute_reward(
                    sinr_db, beam_changed)

                if prev_state is not None:
                    agent.buffer.push(
                        prev_state, prev_beam,
                        reward, state, False)

                loss = agent.train_step()
                ep_rewards.append(reward)
                sinrs.append(sinr_db)
                if beam_changed:
                    switches.append(1)
                if loss > 0:
                    losses.append(loss)

                prev_state = state.copy()
                prev_beam  = action
                step      += 1

        except Exception as e:
            if "Broken pipe" not in str(e):
                print(f"\nEp {ep}: {e}")
        finally:
            try:
                state_f.close()
                act_f.close()
            except Exception:
                pass
            ns3_proc.terminate()
            ns3_proc.wait()

        agent.episode = ep

        # Manual epsilon decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = max(
                agent.epsilon_min,
                agent.epsilon * 0.995)

        ep_r   = float(np.sum(ep_rewards)) \
                 if ep_rewards else 0.0
        sinr_m = float(np.mean(sinrs)) \
                 if sinrs else 0.0
        sw_r   = len(switches) / max(
                     len(ep_rewards), 1)
        loss_m = float(np.mean(losses)) \
                 if losses else 0.0

        log_f.write(
            f"{ep},{ep_r:.3f},{sinr_m:.2f},"
            f"{sw_r:.3f},{loss_m:.5f},"
            f"{agent.epsilon:.4f}\n")
        log_f.flush()

        pbar.set_postfix(
            R=f"{ep_r:.1f}",
            SINR=f"{sinr_m:.1f}",
            eps=f"{agent.epsilon:.3f}")

        if ep % 200 == 0 and ep > start_ep:
            agent.save(str(
                ckpt_dir /
                f"dqn_ep{ep:04d}.pt"))

    agent.save(str(ckpt_dir / "dqn_final.pt"))
    log_f.close()
    print("\nResume training complete.")


if __name__ == "__main__":
    main()

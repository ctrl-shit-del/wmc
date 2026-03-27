# Intelligent Beam Management for 5G/6G mmWave Networks
## Using Deep Reinforcement Learning — VIT Chennai B.Tech ECE 2026

**Team:** Harsh Singhal + Team | Guide: Dr. Hemanth C

---

## Quick Summary
DQN agent trained online in NS-3/5G-LENA (28 GHz mmWave).
- 2000 episodes: 500 exploration + 1500 exploitation
- Beam switch rate: 87% → <2% (98% reduction)
- SINR: statistically equivalent to P1 sweep (p = 0.99)
- Inference latency: 0.01 ms (100× faster than TTI budget)

## Repo Structure
```
beam_project/
├── agent/               Python DRL agent (DQN, PER, training loops)
├── checkpoints/         Trained model weights (.pt files)
├── evaluation/          Frozen policy evaluation + Welch t-test
├── logs/                Training CSV logs + tqdm output
├── plots/               Convergence and comparison figures
├── results/baseline/    NS-3 baseline SINR CSVs (5 seeds × 7 files)
├── ns3_src/             Custom NS-3 C++ simulation files only
│                        (full NS-3+5G-LENA build not included — see VERSION.md)
├── config.json          All hyperparameters
├── version_lock.yaml    Exact NS-3 + 5G-LENA git hashes
└── docs/                Project report and implementation guidelines
```

## Setup
See `ns3_src/VERSION.md` for NS-3 build instructions.
```bash
python3 -m venv ~/beam_drl_env
source ~/beam_drl_env/bin/activate
pip install torch==2.10.0+cpu numpy==2.2.6 scipy matplotlib tqdm
```

## Run
```bash
# Evaluation only (model already trained)
source ~/beam_drl_env/bin/activate
python3 evaluation/evaluate.py
```

## Key Files
| File | Purpose |
|------|---------|
| `agent/dqn_network.py` | DQN + action masking (177K params) |
| `agent/dqn_agent.py` | Reward, select_action, train_step |
| `agent/train.py` | Phase 1: 500 exploration episodes |
| `agent/resume_train.py` | Phase 2: 1500 exploitation episodes |
| `checkpoints/dqn_final.pt` | Final trained model |
| `logs/training_log.csv` | Full 2000-episode training log |
| `ns3_src/drl_beam_sim.cc` | NS-3 simulation with named-pipe IPC |

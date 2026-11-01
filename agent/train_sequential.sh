#!/bin/bash
source ~/beam_drl_env/bin/activate

echo "========================================"
echo "Starting Scenario 1..."
python3 train_multiue.py --scenario 1 --episodes 50 --simtime 3 > ../logs/seq_train_s1.log 2>&1
echo "Scenario 1 Complete."

echo "========================================"
echo "Starting Scenario 2..."
python3 train_multiue.py --scenario 2 --episodes 50 --simtime 3 > ../logs/seq_train_s2.log 2>&1
echo "Scenario 2 Complete."

echo "========================================"
echo "Starting Scenario 3..."
python3 train_multiue.py --scenario 3 --episodes 50 --simtime 3 > ../logs/seq_train_s3.log 2>&1
echo "Scenario 3 Complete."

echo "========================================"
echo "Starting Scenario 4..."
python3 train_multiue.py --scenario 4 --episodes 50 --simtime 3 > ../logs/seq_train_s4.log 2>&1
echo "Scenario 4 Complete."

echo "========================================"
echo "All sequential training successfully completed!"

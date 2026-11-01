import subprocess
import sys

def main():
    episodes = 50
    simtime = 3
    
    print(f"Starting Parallel Training for Scenarios 1-4...")
    print(f"Episodes: {episodes} | SimTime: {simtime}s")
    print("This will heavily utilize your CPU. Please ensure your laptop is plugged in.\n")
    
    processes = []
    
    for s in [1, 2, 3, 4]:
        # Open a separate log file for each scenario so they don't overwrite the terminal
        log_file = open(f"../logs/parallel_train_s{s}.log", "w")
        
        cmd = [
            "python3", "train_multiue.py",
            "--scenario", str(s),
            "--episodes", str(episodes),
            "--simtime", str(simtime)
        ]
        
        # Launch the process in the background
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((s, p, log_file))
        print(f"Launched Scenario {s} (PID: {p.pid}) -> Logging to logs/parallel_train_s{s}.log")
        
    print("\nAll 4 scenarios are now training simultaneously!")
    print("You can check progress in another terminal using: tail -f ../logs/parallel_train_s1.log")
    
    # Wait for all processes to finish
    for s, p, log_file in processes:
        p.wait()
        log_file.close()
        print(f"Scenario {s} has finished training!")
        
    print("\n=== Parallel Training Complete ===")

if __name__ == "__main__":
    main()

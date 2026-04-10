import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Pioneer FDM: Global High-Fidelity Flight Launcher")
    parser.add_argument("--mode", type=str, choices=["sequential", "stochastic"], default="sequential", help="Marathon simulation mode")
    parser.add_argument("--aircraft", type=int, default=5, help="Number of concurrent aircraft")
    args = parser.parse_args()

    print(f"Pioneer FDM | Mode: {args.mode} | Scaling: {args.aircraft} aircraft")
    
    # Dynamically locate the marathon logic from the library
    try:
        from examples.model_test_run import run_marathon
    except ImportError:
        print("ERROR: Could not locate the marathon logic. Ensure warp_jsb is installed correctly.")
        sys.exit(1)
        
    run_marathon(mode=args.mode, num_aircraft=args.aircraft)

if __name__ == "__main__":
    main()

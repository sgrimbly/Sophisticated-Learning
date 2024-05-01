import argparse
import sys
sys.path.append('algorithms')
from algorithms.sophisticated_inference.sophisticated_inference import experiment as si_experiment

def main(algorithm, seed):
    if algorithm == "SI":
        si_experiment(seed)
    # elif algorithm == "SL":
    #     sl_experiment(seed)
    # elif algorithm == "BA":
    #     ba_experiment(seed)
    # elif algorithm == "BAUCB":
    #     baucb_experiment(seed)
    else:
        raise ValueError(f"Invalid algorithm type: {algorithm}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with a specific random seed and algorithm type.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--algorithm', type=str, default='SI', choices=['SI', 'SL', 'BA', 'BAUCB'], help='Type of algorithm to run.')
    args = parser.parse_args()

    main(algorithm=args.algorithm, seed=args.seed)

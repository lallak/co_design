import sys
import argparse
import numpy as np
from hopper_codesign.optimize import run_codesign
from hopper_codesign.visualize_hopper import plot_convergence, render_best_design
from hopper_codesign.evaluator import evaluate_design, sensitivity_analysis
from hopper_codesign.evaluator import debug_controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hopper Co-Design Experiment")
    parser.add_argument('mode', choices=['optimize', 'visualize', 'render', 'test', 'sensitivity'],
                        help="Mode to run")
    parser.add_argument('--theta', nargs=3, type=float,
                        default=[0.35, 0.4, 2.86],
                        help="Design parameters [thigh_length, leg_length, rho]")
    args = parser.parse_args()

    if args.mode == 'optimize':
        print("Running full co-design optimization...")
        theta_opt, history = run_codesign()
        print("\nGenerating convergence plots...")
        plot_convergence()
        print("\nRendering best design...")
        render_best_design(theta_opt)

    elif args.mode == 'visualize':
        print("Plotting convergence from saved results...")
        plot_convergence()

    elif args.mode == 'render':
        theta = np.array(args.theta)
        render_best_design(theta)

    elif args.mode == 'test':
        # Quick sanity check: evaluate the default design
        theta = np.array(args.theta)
        print(f"Testing design θ={np.round(theta, 3)}")
        reward = evaluate_design(theta, seed=0)
        print(f"✅ Design θ={np.round(theta, 3)} → reward={reward:.2f}")

    elif args.mode == 'sensitivity':
        theta = np.array(args.theta)
        print(f"Running sensitivity analysis around θ={np.round(theta, 3)}")
        results = sensitivity_analysis(theta)
        print("\nRanked by sensitivity:")
        for name, val in sorted(results.items(), key=lambda x: -x[1]):
            bar = '█' * int(val * 20)
            print(f"  {name:>15}: {val:.3f}  {bar}")

    elif args.mode == 'debug':
        theta = np.array(args.theta)
        print(f"Debugging controller for θ={np.round(theta, 3)}")
        debug_controller(theta)


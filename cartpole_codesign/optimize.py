import numpy as np
import cma #bc impossible to backpropagate through MPPI (obj function=black box, small design space, optimization space=non convex + noisy)
import json
from cartpole_codesign.evaluator import evaluate_design

MODEL_NAME = "cartpole"
CONTROLLER_NAME = "MPPI"

def run_codesign(output_path: str = "results_cartpole/optimization_log.json"):
    """
    Run CMA-ES to find the optimal CartPole design for swingup.
    Optimizes: [pole_length, cart_mass]
    Fixed: pole_mass = 0.1 kg
    """

    # --- Design parameter bounds and initial guess ---
    # theta = [pole_length, cart_mass]
    theta0 = [0.8, 1.0]  # initial guess
    sigma0 = 0.1  # initial step size (spread of search)

    lower_bounds = [0.3, 0.5]   # pole_length, cart_mass
    upper_bounds = [1.5, 3.0]   # pole_length, cart_mass

    # --- CMA-ES options ---
    options = {
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': 10,  # number of generations
        'popsize': 5,  # candidates per generation (λ)
        'verbose': 1,  # print progress
        'seed': 42,
    }

    es = cma.CMAEvolutionStrategy(theta0, sigma0, options)

    # --- Logging ---
    history = {
        'generations': [],
        'best_fitness': [],
        'best_theta': [],
        'all_fitnesses': [],
        'all_candidates': [],
    }

    generation = 0
    max_gen = options["maxiter"]

    print("\n==============================")
    print(f"MODEL: {MODEL_NAME}")
    print(f"CONTROLLER: {CONTROLLER_NAME}")
    print(f"MAX GENERATIONS: {max_gen}")
    print("==============================\n")

    # --- Optimization loop ---
    while not es.stop():
        # Ask for a new batch of candidate designs
        candidates = es.ask()  # list of 12 design vectors

        print(f"\n=== Generation {generation} ===")
        print(f"Evaluating {len(candidates)} candidates...")

        # Evaluate each candidate (this is the expensive step)
        fitnesses = [] #list of scores
        for i, theta in enumerate(candidates):
            fitness = evaluate_design(np.array(theta), seed=generation * 100 + i) #function from evaluator
            fitnesses.append(fitness)
            print(f"  Candidate {i:2d}: θ={np.round(theta, 3)}, reward={fitness:.2f}")

        # CMA-ES minimizes, so we negate the reward
        es.tell(candidates, [-f for f in fitnesses]) #remembers good candidates and their scores to update the search distribution

        # Log results_cartpole
        best_idx = np.argmax(fitnesses)
        history['generations'].append(generation)
        history['best_fitness'].append(float(fitnesses[best_idx]))
        history['best_theta'].append(candidates[best_idx].tolist())
        history['all_fitnesses'].append([float(f) for f in fitnesses])
        history['all_candidates'].append([c.tolist() for c in candidates])

        print(f"  Best this generation: reward={fitnesses[best_idx]:.2f}, "
              f"θ={np.round(candidates[best_idx], 3)}")

        generation += 1

    # --- Report results_cartpole ---
    theta_opt = es.result.xbest
    print(f"\n=== Optimization Complete ===")
    print(f"Optimal design: {np.round(theta_opt, 4)}")
    print(f"  pole_length = {theta_opt[0]:.3f} m")
    print(f"  cart_mass   = {theta_opt[1]:.3f} kg")
    print(f"  pole_mass   = 0.100 kg (fixed)")

    # Save history
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)

    return theta_opt, history


if __name__ == "__main__":
    theta_opt, history = run_codesign()
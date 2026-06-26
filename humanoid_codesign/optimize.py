import numpy as np
import cma
import json
from hopper_codesign.evaluator import evaluate_design


def run_codesign(output_path: str = "results/results_hopper/hopper_optimization_log.json"):
    """Run CMA-ES optimization for hopper design."""

    theta0 = [0.35, 0.4, 2.0]  # [thigh_length, leg_length, rho]
    sigma0 = 0.1

    lower_bounds = [0.05, 0.05, 0.1] #same bounds as the ones defined in the model builder 
    upper_bounds = [0.4, 0.4, 5.0]
    
    options = {
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': 30,
        'popsize': 12,
        'verbose': 1,
        'seed': 42,
    }
    
    es = cma.CMAEvolutionStrategy(theta0, sigma0, options)
    
    history = {
        'generations': [],
        'best_fitness': [],
        'best_theta': [],
        'all_fitnesses': [],
        'all_candidates': [],
    }
    
    generation = 0
    
    while not es.stop():
        candidates = es.ask()
        print(f"\n=== Generation {generation} ===")
        print(f"Evaluating {len(candidates)} candidates...")
        
        fitnesses = []
        for i, theta in enumerate(candidates):
            fitness = evaluate_design(np.array(theta), seed=generation * 100 + i)
            fitnesses.append(fitness)
            print(f"  Candidate {i:2d}: reward={fitness:.2f}")
        
        es.tell(candidates, [-f for f in fitnesses])
        
        best_idx = np.argmax(fitnesses)
        history['generations'].append(generation)
        history['best_fitness'].append(float(fitnesses[best_idx]))
        history['best_theta'].append(candidates[best_idx].tolist())
        history['all_fitnesses'].append([float(f) for f in fitnesses])
        history['all_candidates'].append([c.tolist() for c in candidates])
        
        print(f"  Best: reward={fitnesses[best_idx]:.2f}")
        generation += 1
    
    theta_opt = es.result.xbest
    print(f"\n=== Optimization Complete ===")
    print(f"Best design: {np.round(theta_opt, 3)}")
    
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("Stop conditions:", es.stop()) #to understand why it doesnt go further than gen 0
    
    return theta_opt, history


if __name__ == "__main__":
    theta_opt, history = run_codesign()


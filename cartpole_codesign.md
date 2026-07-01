  # Co-Design of CartPole via Sampling-Based MPC

## Overview

This document describes a co-design framework that jointly optimizes the **physical design parameters** and **control behavior** of a CartPole system using sampling-based Model Predictive Control (MPC). It is intended as both a research implementation guide and a learning resource for students new to co-design.

---

## 1. Background and Motivation

### What is Co-Design?

Traditional robot development separates two concerns:
1. **Morphology design** — choosing the physical parameters of the robot (lengths, masses, stiffnesses)
2. **Control design** — developing a controller for a fixed robot

This separation is suboptimal. A longer pole may be easier to balance but harder to swing up. A heavier cart may be more stable but less responsive. **The best robot design depends on the controller, and the best controller depends on the robot design.** Co-design optimizes both together.

### Why Sampling-Based MPC as the Inner Loop?

Most existing co-design frameworks use one of two approaches for the inner loop (evaluating how good a design is):

- **Reinforcement Learning (RL)**: Trains a policy from scratch for each candidate design. This works but is very slow — training even a simple policy can take thousands of environment steps, and you need to do this for every design candidate the optimizer proposes.
- **Differentiable Simulation**: Backpropagates gradients through the simulator directly into the design parameters. This is fast but only works for smooth, differentiable dynamics. It fails for contact-rich systems (impacts, friction).

**Sampling-based MPC offers a middle path:**
- It finds near-optimal control behavior for any fixed design **without training** — just by rolling out many trajectories in simulation and picking the best
- It works on non-smooth dynamics (important for future extensions to contact-rich systems)
- On GPU with JAX/MJX, it evaluates thousands of rollouts in parallel, making it fast enough to use as an inner-loop evaluator

### Why CartPole First?

CartPole is an ideal starting system because:
- The dynamics are well-understood and simple to implement
- The design space is small and interpretable (2–3 parameters)
- Sampling-based MPC (specifically MPPI) is known to solve CartPole swingup well — there are existing Hydrax examples to build on
- It lets us validate the full pipeline before moving to contact-rich systems like hoppers

---

## 2. Problem Formulation

### System Description

A CartPole consists of:
- A **cart** that slides horizontally on a frictionless track
- A **pole** attached to the cart via a passive revolute joint
- A **single actuator**: a horizontal force applied to the cart

The state vector is `x = [cart_position, pole_angle, cart_velocity, pole_angular_velocity]` ∈ ℝ⁴.

The control input is `u = [cart_force]` ∈ ℝ¹.

### Design Parameters

We optimize over a design vector **θ ∈ ℝ³**:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Pole length | `L` | [0.3, 1.5] m | Length of the pole |
| Pole mass | `m_p` | [0.05, 0.5] kg | Mass of the pole |
| Cart mass | `m_c` | [0.5, 3.0] kg | Mass of the cart |

These parameters affect the natural frequency of the system, how much force is needed for swingup, and the stability margin near the inverted equilibrium.

### Task

The task is **CartPole swingup**: starting from the pole hanging downward (stable equilibrium), swing the pole up to the inverted position and balance it there.

**Why swingup (not just balancing)?** Balancing near the top is a linear control problem that any design can solve. Swingup requires exploiting nonlinear dynamics and is sensitive to the design — a too-heavy or too-short pole may not be swingable within the cart's track limits.

### Cost Function

The running cost encourages the pole to stay upright and the cart to stay near center:

```
ℓ(x, u) = (1 - cos(θ_pole)) + 0.1 * x_cart² + 0.01 * u²
```

- `(1 - cos(θ_pole))`: zero when pole is upright, 2 when hanging — a smooth measure of pole angle error
- `0.1 * x_cart²`: keeps the cart near the center of the track
- `0.01 * u²`: penalizes large control forces (energy efficiency)

The terminal cost uses the same form evaluated at the final state.

### Optimization Objective

Find the design θ* that maximizes the performance of the best achievable controller:

```
θ* = argmax_θ  J(θ)

where J(θ) = (1/N) Σ_{i=1}^{N} R_i(θ, π_MPC(θ))
```

- `N` = number of evaluation episodes
- `R_i` = cumulative reward in episode `i`  
- `π_MPC(θ)` = the MPC policy running on a robot with design `θ`

The key insight: we don't optimize over a fixed policy class — we let MPC find the best possible behavior for each θ, so J(θ) reflects the **design's intrinsic performance potential**.

---

## 3. Software Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Physics simulation | MuJoCo + MJX | Fast GPU-parallel dynamics |
| Sampling-based MPC | Hydrax | MPPI / Predictive Sampling controller |
| Automatic differentiation | JAX | Vectorized rollouts via `vmap` |
| Design optimization | PyCMA (CMA-ES) | Black-box outer-loop optimizer |
| Visualization | Hydrax interactive viewer | Render best design |

### Installing Dependencies

```bash
# Install uv (recommended, following Hydrax docs)
pip install uv

# Clone and install Hydrax
git clone https://github.com/vincekurtz/hydrax.git
cd hydrax
uv sync

# Install additional dependencies
pip install cma matplotlib
```

---

## 4. Implementation Steps

### Step 1: Create a Parameterized MuJoCo Model

**File:** `cartpole_codesign/assets/cartpole_template.xml`

The MuJoCo model must be parameterizable — we need to instantiate a different model for each design candidate θ. The cleanest approach is to use a Python string template with `{placeholders}` for the design parameters.

```xml
<mujoco model="cartpole_codesign">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  
  <worldbody>
    <!-- Cart -->
    <body name="cart" pos="0 0 0.05">
      <joint name="slider" type="slide" axis="1 0 0" range="-2.4 2.4"/>
      <geom type="box" size="0.15 0.08 0.05" mass="{cart_mass}"/>
      
      <!-- Pole -->
      <body name="pole" pos="0 0 0.05">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 {pole_length}" 
              size="0.02" mass="{pole_mass}"/>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="slide" joint="slider" gear="1" ctrllimit="-10 10"/>
  </actuator>
</mujoco>
```

**Python function to instantiate a model from θ:**

```python
import mujoco
import numpy as np

TEMPLATE_PATH = "cartpole_codesign/assets/cartpole_template.xml"

def build_cartpole_model(theta: np.ndarray) -> mujoco.MjModel:
    """
    Build a MuJoCo model from design parameters.
    
    Args:
        theta: [pole_length, pole_mass, cart_mass]
    
    Returns:
        mj_model: MuJoCo model with the specified design
    """
    pole_length, pole_mass, cart_mass = theta
    
    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()
    
    xml_string = template.format(
        pole_length=pole_length,
        pole_mass=pole_mass,
        cart_mass=cart_mass
    )
    
    return mujoco.MjModel.from_xml_string(xml_string)
```

> **Student note:** `mujoco.MjModel.from_xml_string()` loads a model directly from a string without writing to disk. This is important because the outer optimizer will call `build_cartpole_model()` hundreds of times — we don't want to create hundreds of XML files.

---

### Step 2: Define the Hydrax Task

**File:** `cartpole_codesign/tasks/cartpole_task.py`

In Hydrax, a `Task` defines the cost function and wraps a MuJoCo model. We subclass `hydrax.task_base.Task`:

```python
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.task_base import Task


class CartpoleSwingupTask(Task):
    """
    CartPole swingup task for co-design.
    
    The pole starts hanging down. The goal is to swing it up
    and balance it at the inverted position.
    """
    
    def __init__(self, mj_model: mujoco.MjModel, 
                 plan_horizon: float = 1.0,
                 num_knots: int = 10,
                 dt: float = 0.01):
        """
        Args:
            mj_model:      MuJoCo model (built from design parameters)
            plan_horizon:  How far ahead to plan, in seconds
            num_knots:     Number of spline control points
            dt:            Simulation timestep (should match XML)
        """
        super().__init__(mj_model, plan_horizon=plan_horizon,
                        num_knots=num_knots, dt=dt)
    
    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        """
        Cost at each timestep. Lower is better.
        
        State layout for CartPole in MuJoCo:
            x.qpos[0] = cart position
            x.qpos[1] = pole angle (0 = hanging down, π = upright)
            x.qvel[0] = cart velocity  
            x.qvel[1] = pole angular velocity
        """
        cart_pos = x.qpos[0]
        pole_angle = x.qpos[1]
        
        # (1 - cos) is 0 when upright (angle=π), 2 when hanging (angle=0)
        # We use pole_angle directly — check your model's angle convention
        upright_cost = 1.0 - jnp.cos(pole_angle)
        
        # Keep cart near center
        cart_cost = 0.1 * jnp.square(cart_pos)
        
        # Penalize large control inputs
        control_cost = 0.01 * jnp.sum(jnp.square(u))
        
        return upright_cost + cart_cost + control_cost
    
    def terminal_cost(self, x: mjx.Data) -> float:
        """Cost at the final state (end of planning horizon)."""
        cart_pos = x.qpos[0]
        pole_angle = x.qpos[1]
        
        return (1.0 - jnp.cos(pole_angle)) + 0.1 * jnp.square(cart_pos)
```

> **Student note:** The `running_cost` function is called inside JAX's JIT compiler, so it must use `jax.numpy` (imported as `jnp`) instead of regular `numpy`. Regular Python control flow (if/else) is fine, but NumPy operations will cause errors.

---

### Step 3: Write the Design Evaluator

**File:** `cartpole_codesign/evaluator.py`

This is the **inner loop** — it takes a design θ and returns a scalar fitness score by running MPC on the CartPole.

```python
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.algs import MPPI

from cartpole_codesign.tasks.cartpole_task import CartpoleSwingupTask
from cartpole_codesign.assets.model_builder import build_cartpole_model


# Hyperparameters (fixed across all designs)
PLAN_HORIZON = 1.0      # seconds
NUM_KNOTS    = 10       # spline control points
DT           = 0.01     # simulation timestep
NUM_SAMPLES  = 256      # MPPI trajectory samples per step
NOISE_LEVEL  = 0.5      # MPPI exploration noise
EPISODE_STEPS = 400     # steps per episode (~4 seconds)
NUM_EPISODES  = 3       # episodes per design evaluation


def evaluate_design(theta: np.ndarray, seed: int = 0) -> float:
    """
    Evaluate a design by running MPPI on the CartPole swingup task.
    
    Args:
        theta: Design vector [pole_length, pole_mass, cart_mass]
        seed:  Random seed for reproducibility
    
    Returns:
        mean_reward: Average cumulative reward across NUM_EPISODES episodes.
                     Higher is better (we negate this for CMA-ES which minimizes).
    """
    # --- Build model and task ---
    mj_model = build_cartpole_model(theta)
    task = CartpoleSwingupTask(mj_model, 
                               plan_horizon=PLAN_HORIZON,
                               num_knots=NUM_KNOTS, 
                               dt=DT)
    
    # --- Initialize MPPI controller ---
    controller = MPPI(task, 
                      num_samples=NUM_SAMPLES,
                      noise_level=NOISE_LEVEL)
    
    # --- JIT-compile the MPC update step (do this once per design) ---
    jit_update = jax.jit(controller.update)
    jit_get_action = jax.jit(controller.get_action)
    
    # --- Run episodes ---
    rng = jax.random.PRNGKey(seed)
    total_reward = 0.0
    
    for episode in range(NUM_EPISODES):
        # Reset to initial state: pole hanging down, small random perturbation
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[1] = np.pi + np.random.uniform(-0.1, 0.1)  # pole down
        mj_data.qvel[:] = 0.0
        
        # Convert to MJX data for JAX operations
        mjx_data = mjx.put_data(mj_model, mj_data)
        
        # Initialize controller state
        rng, subkey = jax.random.split(rng)
        ctrl_state = controller.init(subkey)
        
        episode_reward = 0.0
        
        for step in range(EPISODE_STEPS):
            # MPC update: sample trajectories and compute optimal action
            rng, subkey = jax.random.split(rng)
            ctrl_state = jit_update(ctrl_state, mjx_data, subkey)
            
            # Get the first action from the optimal plan
            action = jit_get_action(ctrl_state)
            
            # Apply action and step simulation
            mj_data.ctrl[:] = np.array(action)
            mujoco.mj_step(mj_model, mj_data)
            mjx_data = mjx.put_data(mj_model, mj_data)
            
            # Accumulate reward (negative cost)
            cost = task.running_cost(mjx_data, action)
            episode_reward -= float(cost)
        
        total_reward += episode_reward
    
    return total_reward / NUM_EPISODES
```

> **Student note:** Notice that `jax.jit` is called **once per design**, not once per step. This is important — JIT compilation traces the function the first time it's called and caches the compiled version. If you called `jax.jit` inside the step loop, it would recompile every step, which is very slow.

---

### Step 4: Run the CMA-ES Outer Loop

**File:** `cartpole_codesign/optimize.py`

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a black-box optimizer for continuous parameters. It maintains a probability distribution over the design space and updates it each generation based on which candidates performed best.

```python
import numpy as np
import cma
import json
from cartpole_codesign.evaluator import evaluate_design


def run_codesign(output_path: str = "results_cartpole/optimization_log.json"):
    """
    Run CMA-ES to find the optimal CartPole design for swingup.
    """
    
    # --- Design parameter bounds and initial guess ---
    # theta = [pole_length, pole_mass, cart_mass]
    theta0    = [0.8,  0.1,  1.0]   # initial guess (reasonable default)
    sigma0    = 0.3                  # initial step size (spread of search)
    
    lower_bounds = [0.3, 0.05, 0.5]
    upper_bounds = [1.5, 0.50, 3.0]
    
    # --- CMA-ES options ---
    options = {
        'bounds':   [lower_bounds, upper_bounds],
        'maxiter':  30,        # number of generations
        'popsize':  12,        # candidates per generation (λ)
        'verbose':  1,         # print progress
        'seed':     42,
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
    
    # --- Optimization loop ---
    while not es.stop():
        # Ask for a new batch of candidate designs
        candidates = es.ask()   # list of 12 design vectors
        
        print(f"\n=== Generation {generation} ===")
        print(f"Evaluating {len(candidates)} candidates...")
        
        # Evaluate each candidate (this is the expensive step)
        fitnesses = []
        for i, theta in enumerate(candidates):
            fitness = evaluate_design(np.array(theta), seed=generation * 100 + i)
            fitnesses.append(fitness)
            print(f"  Candidate {i:2d}: θ={np.round(theta, 3)}, reward={fitness:.2f}")
        
        # CMA-ES minimizes, so we negate the reward
        es.tell(candidates, [-f for f in fitnesses])
        
        # Log results_cartpole
        best_idx = np.argmax(fitnesses)
        history['generations'].append(generation)
        history['best_fitness'].append(fitnesses[best_idx])
        history['best_theta'].append(candidates[best_idx])
        history['all_fitnesses'].append(fitnesses)
        history['all_candidates'].append([c.tolist() for c in candidates])
        
        print(f"  Best this generation: reward={fitnesses[best_idx]:.2f}, "
              f"θ={np.round(candidates[best_idx], 3)}")
        
        generation += 1
    
    # --- Report results_cartpole ---
    theta_opt = es.result.xbest
    print(f"\n=== Optimization Complete ===")
    print(f"Optimal design: {np.round(theta_opt, 4)}")
    print(f"  pole_length = {theta_opt[0]:.3f} m")
    print(f"  pole_mass   = {theta_opt[1]:.3f} kg")
    print(f"  cart_mass   = {theta_opt[2]:.3f} kg")
    
    # Save history
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return theta_opt, history


if __name__ == "__main__":
    theta_opt, history = run_codesign()
```

> **Student note:** CMA-ES maintains a multivariate Gaussian distribution over the design space. `es.ask()` samples `popsize` candidates from this distribution. `es.tell()` updates the distribution based on which candidates had lower cost. Over generations, the distribution converges toward the region of design space with the best performance.

---

### Step 5: Visualize Results

**File:** `cartpole_codesign/visualize.py`

Two types of visualization: convergence plots and an interactive simulation of the best design.

```python
import numpy as np
import matplotlib.pyplot as plt
import json
from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive

from cartpole_codesign.tasks.cartpole_task import CartpoleSwingupTask
from cartpole_codesign.assets.model_builder import build_cartpole_model


def plot_convergence(history_path: str = "results_cartpole/optimization_log.json"):
    """Plot best fitness and design parameters over generations."""
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    generations  = history['generations']
    best_fitness = history['best_fitness']
    best_thetas  = np.array(history['best_theta'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CartPole Co-Design: CMA-ES Convergence", fontsize=14)
    
    # Fitness over generations
    axes[0, 0].plot(generations, best_fitness, 'b-o', linewidth=2)
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Best Reward")
    axes[0, 0].set_title("Best Design Performance per Generation")
    axes[0, 0].grid(True)
    
    # Design parameter evolution
    param_names = ['Pole Length (m)', 'Pole Mass (kg)', 'Cart Mass (kg)']
    colors = ['r', 'g', 'b']
    
    for i, (name, color) in enumerate(zip(param_names, colors)):
        ax = axes[(i + 1) // 2, (i + 1) % 2]
        ax.plot(generations, best_thetas[:, i], f'{color}-o', linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel(name)
        ax.set_title(f"Optimal {name} over Generations")
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("results_cartpole/convergence_30_12_slowmode.png", dpi=150)
    plt.show()
    print("Saved convergence plot to results_cartpole/convergence_30_12_slowmode.png")


def render_best_design(theta: np.ndarray):
    """
    Render an interactive simulation of the best design using Hydrax viewer.
    
    Args:
        theta: Optimal design [pole_length, pole_mass, cart_mass]
    """
    print(f"Rendering design: pole_length={theta[0]:.3f}, "
          f"pole_mass={theta[1]:.3f}, cart_mass={theta[2]:.3f}")
    
    mj_model = build_cartpole_model(theta)
    task = CartpoleSwingupTask(mj_model, plan_horizon=1.0, num_knots=10, dt=0.01)
    controller = MPPI(task, num_samples=512, noise_level=0.5)
    
    # run_interactive opens a MuJoCo viewer window
    run_interactive(task, controller)
```

---

### Step 6: Entry Point

**File:** `cartpole_codesign/run_experiment.py`

```python
import argparse
import numpy as np
from cartpole_codesign.optimize import run_codesign
from cartpole_codesign.visualize import plot_convergence, render_best_design
from cartpole_codesign.evaluator import evaluate_design


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole Co-Design Experiment")
    parser.add_argument('mode', choices=['optimize', 'visualize', 'render', 'test'],
                        help="Mode to run")
    parser.add_argument('--theta', nargs=3, type=float,
                        default=[0.8, 0.1, 1.0],
                        help="Design parameters for render/test mode")
    args = parser.parse_args()
    
    if args.mode == 'optimize':
        theta_opt, history = run_codesign()
        plot_convergence()
        render_best_design(theta_opt)
    
    elif args.mode == 'visualize':
        plot_convergence()
    
    elif args.mode == 'render':
        theta = np.array(args.theta)
        render_best_design(theta)
    
    elif args.mode == 'test':
        # Quick sanity check: evaluate the default design
        theta = np.array(args.theta)
        reward = evaluate_design(theta, seed=0)
        print(f"Design θ={theta} → reward={reward:.2f}")
```

**Usage:**
```bash
# Run full co-design optimization
python3 -m cartpole_codesign.run_experiment optimize

# Test a specific design
python3 -m cartpole_codesign.run_experiment test --theta 0.8 0.1 1.0

# Render the best found design interactively
python3 -m cartpole_codesign.run_experiment render --theta 1.1 0.15 1.5

# Plot convergence from saved results_cartpole
python3 -m cartpole_codesign.run_experiment visualize
```

---

## 5. Project File Structure

```
cartpole_codesign/
├── assets/
│   ├── cartpole_template.xml      # Parameterized MuJoCo model
│   └── model_builder.py           # build_cartpole_model(theta)
├── tasks/
│   └── cartpole_task.py           # CartpoleSwingupTask(Task)
├── evaluator.py                   # evaluate_design(theta) → float
├── optimize.py                    # CMA-ES outer loop
├── visualize.py                   # Convergence plots + interactive render
└── run_experiment.py              # Entry point (CLI)
results/
├── optimization_log.json          # Saved per-generation data
└── convergence.png                # Convergence plot
```

---

## 6. Expected Results and Interpretation

### What Should You See?

After ~30 generations (360 total evaluations), CMA-ES should converge to a design with measurably better swingup performance than the default. Typical findings for CartPole swingup:

- **Longer poles** tend to perform better because they have more gravitational potential energy to exploit during the swing, and are easier to balance once upright (slower natural frequency)
- **Lighter poles** relative to cart mass allow more responsive swinging without saturating the actuator
- **Heavier carts** provide a more stable base, reducing unwanted cart motion during the swing

### Sanity Checks

Before running the full optimization, verify each component works:

1. **Model builds correctly**: `python -m cartpole_codesign.run_experiment test --theta 0.8 0.1 1.0` should print a reward without errors
2. **MPC solves the task**: render the default design and verify the pole swings up within ~5 seconds
3. **CMA-ES evaluates a single generation**: modify `maxiter=1` and confirm 12 candidates are evaluated and logged

### Convergence Diagnostics

- If best fitness **doesn't improve after 10 generations**: the reward signal may be too noisy — increase `NUM_EPISODES` from 3 to 5
- If all designs **converge to the boundary** of the design space: the bounds may be too restrictive or the initial sigma too small — increase `sigma0`
- If **reward is always near zero**: the MPC may not be solving the swingup — increase `NUM_SAMPLES` or `PLAN_HORIZON`

---

## 7. Extensions

Once the basic pipeline works, consider these extensions ordered by difficulty:

1. **Expand the design space** — add pole width, actuator force limit, or track length as additional design parameters
2. **Multi-task co-design** — evaluate each design on both swingup *and* a disturbance rejection task; take the average as fitness
3. **Baseline comparison** — implement a simple RL inner loop (PPO via Brax) and compare the number of simulation steps needed to reach the same design quality
4. **Planar hopper** — replace the CartPole model with a hopping robot; the same CMA-ES outer loop and evaluator structure apply directly, but now contact dynamics matter — this is the key motivation for using sampling-based MPC over differentiable simulation

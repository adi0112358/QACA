import os
from datetime import datetime
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from environment.environment import GridWorld

from agent.state_model import StateModel
from agent.world_model import WorldModel
from agent.value_model import ValueModel
from agent.meta_model import MetaStateModel

from planning.cem_planner import CEMPlanner
from governance.constraints import GovernanceConfig
from governance.divergence import DivergenceConfig, BarrierConfig, compute_divergence, compute_barriers
from governance.controller import GovernanceController, GovernanceControllerConfig
from governance.budgeting import RegionalBudgetAllocator

from consistency.mixed_consistency import MixedConsistencyLayer
from epl import parse_epl, EPLRuntime, EPLRuntimeConfig, compile_to_automaton, HybridCompileConfig

from training.replay_buffer import ReplayBuffer
from training.trainer import WorldModelTrainer
from training.value_trainer import ValueTrainer
from training.meta_trainer import MetaTrainer

from utils.heatmap import PredictionErrorHeatmap


# -------------------------
# Environment
# -------------------------

env = GridWorld(
    size=10,
    agent_b_policy="pursue",
    pursuit_prob=0.75,
    perturb_prob=0.03
)
heatmap = PredictionErrorHeatmap(env.size)
budgeter = RegionalBudgetAllocator(env.size)
consistency_layer = MixedConsistencyLayer()

# -------------------------
# Plot Output
# -------------------------

PLOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "plots", "runs")
)
os.makedirs(PLOT_DIR, exist_ok=True)
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
HEADLESS = plt.get_backend().lower().endswith("agg")


def finalize_plot(name):

    if HEADLESS:
        path = os.path.join(PLOT_DIR, f"{RUN_TAG}_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print("Saved plot:", path)
        plt.close()
        return

    plt.show()

# -------------------------
# EPL (Entropy Programming Language)
# -------------------------

epl_runtime = None
epl_enabled = False
hybrid_automaton = None
epl_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "epl", "example.epl")
)

if os.path.exists(epl_path):
    try:
        with open(epl_path, "r") as f:
            epl_program = parse_epl(f.read())
        epl_runtime = EPLRuntime(
            epl_program,
            EPLRuntimeConfig(beta=5.0, default_weight=1.0)
        )
        epl_enabled = True

        hybrid_automaton = compile_to_automaton(
            epl_program,
            HybridCompileConfig(divergence_threshold=1.5, recovery_threshold=0.8)
        )

        for overlay in epl_program.overlays:
            consistency_layer.register_overlay(overlay.name, overlay.consistency)

        if epl_program.budgets:
            budget = epl_program.budgets[0]
            budgeter.set_total_budget(budget.total)

    except Exception as e:
        print("EPL load failed:", e)
        epl_runtime = None
        hybrid_automaton = None


# -------------------------
# Swarm Parameters
# -------------------------

NUM_AGENTS = 1


# -------------------------
# Models
# -------------------------

obs_size = env.size * env.size

state_model = StateModel(obs_size=obs_size)
world_model = WorldModel()
value_model = ValueModel()
meta_model = MetaStateModel()

world_trainer = WorldModelTrainer(world_model)
value_trainer = ValueTrainer(value_model)
meta_trainer = MetaTrainer(meta_model)

governance_config = GovernanceConfig(
    max_state_norm=8.0,
    max_state_delta=3.5,
    max_uncertainty=2.5,
    lyapunov_decay=0.04,
    w_barrier=1.0,
    w_lyapunov=0.6,
    w_uncertainty=0.5,
    w_delta=0.3
)

divergence_config = DivergenceConfig(
    w_error=1.0,
    w_uncertainty=0.6,
    w_value=0.2,
    w_reference=0.4,
    target_error=0.2,
    target_uncertainty=1.0,
    target_value=0.0,
    target_state_norm=3.0
)

barrier_config = BarrierConfig(
    max_state_norm=8.0,
    max_uncertainty=2.5,
    max_error=3.0
)

governance_controller = GovernanceController(
    GovernanceControllerConfig(
        clf_eta=0.25,
        k_v=0.6,
        k_b=1.0,
        u_ref=0.1,
        u_weight=0.4,
        slack_weight=30.0,
        safe_mode_floor=0.85,
        min_horizon=2,
        max_horizon=6,
        min_samples=32,
        max_samples=128,
        min_uncertainty_weight=0.05,
        max_uncertainty_weight=0.3
    )
)

planner = CEMPlanner(
    world_model,
    value_model,
    horizon=4,
    num_samples=64,
    elite_frac=0.2,
    iterations=3,
    alpha=0.7,
    uncertainty_weight=0.1,
    governance_config=governance_config,
    min_horizon=2,
    max_horizon=6,
    min_samples=32,
    max_samples=128
)

buffer = ReplayBuffer()


# -------------------------
# Training Parameters
# -------------------------

episodes = 1
max_steps = 40


# -------------------------
# Plot Storage
# -------------------------

reward_history = []
wm_loss_history = []
prediction_error_history = []
state_trajectory = []


# =========================
# Training Loop
# =========================

for episode in range(episodes):

    print("\n========== Episode", episode, "==========")

    base_epsilon = max(0.02, 0.3 * (0.990 ** episode))
    meta_reliability = 0.5
    meta_risk = 0.5
    gov_exploration_scale = 1.0
    current_mode = "normal"

    states = [state_model.init_state() for _ in range(NUM_AGENTS)]

    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    done = False
    steps = 0
    total_reward = 0


    while not done and steps < max_steps:

        for i in range(NUM_AGENTS):

            state = states[i]

            # ---------------------------------
            # Action Selection
            # ---------------------------------

            epsilon = max(
                0.02,
                base_epsilon * (1.0 - meta_reliability) * gov_exploration_scale
            )

            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = planner.select_action(state)

            action_vec = torch.zeros(1, 4)
            action_vec[0, action] = 1


            # ---------------------------------
            # Predict next state (world model)
            # ---------------------------------

            mu, logvar = world_model(state, action_vec)

            predicted_next_state = mu
            uncertainty = torch.mean(torch.exp(logvar)).item()


            # ---------------------------------
            # Environment step
            # ---------------------------------

            obs_next, reward, done = env.step(action)

            obs_next = torch.tensor(
                obs_next, dtype=torch.float32
            ).unsqueeze(0)


            # ---------------------------------
            # True next state
            # ---------------------------------

            next_state = state_model(obs_next, action_vec, state)


            # ---------------------------------
            # Store experience
            # ---------------------------------

            buffer.push(
                state.detach(),
                action_vec.detach(),
                next_state.detach()
            )


            # ---------------------------------
            # Train World Model
            # ---------------------------------

            if len(buffer) > 32:

                states_batch, actions_batch, next_states_batch = buffer.sample(32)

                wm_loss = world_trainer.train_step(
                    states_batch,
                    actions_batch,
                    next_states_batch
                )

                wm_loss_history.append(wm_loss)

                print("World model loss:", wm_loss)


            # ---------------------------------
            # Prediction Error
            # ---------------------------------

            prediction_error = torch.norm(
                predicted_next_state - next_state
            )

            error_val = prediction_error.item()

            prediction_error_history.append(error_val)

            print("Prediction error:", error_val)


            # ---------------------------------
            # Curiosity Reward
            # ---------------------------------

            lambda_curiosity = 0.05

            intrinsic_reward = lambda_curiosity * error_val

            reward_total = reward + intrinsic_reward

            print("Intrinsic reward:", intrinsic_reward)


            # ---------------------------------
            # Train Value Model
            # ---------------------------------

            reward_tensor = torch.tensor(
                [[reward_total]], dtype=torch.float32
            )

            value_loss = value_trainer.train_step(
                state.squeeze(0).detach(),
                reward_tensor,
                next_state.squeeze(0).detach()
            )

            print("Value loss:", value_loss)


            # ---------------------------------
            # Meta-State Monitoring
            # ---------------------------------

            meta_loss = meta_trainer.train_step(
                state.squeeze(0),
                next_state.squeeze(0),
                error_val,
                uncertainty
            )

            meta_state = meta_model(
                state.squeeze(0),
                error_val,
                uncertainty
            )

            meta_state_np = meta_state.detach().cpu().numpy()[0]
            meta_reliability = float(meta_state_np[0])
            meta_risk = float(meta_state_np[2])

            # ---------------------------------
            # Governance + Budgeting (Entropy Engine)
            # ---------------------------------

            if hasattr(env, "agentA_pos"):
                heatmap.update(env.agentA_pos, error_val)

            with torch.no_grad():
                current_value = value_model(state.squeeze(0)).item()

            state_norm = torch.norm(
                state.squeeze(0).view(1, -1),
                dim=1
            ).item()

            epl_soft_penalty = 0.0
            epl_barriers = {}

            if epl_runtime is not None and epl_enabled:
                try:
                    epl_soft_penalty, epl_barriers = epl_runtime.evaluate(
                        {
                            "error": error_val,
                            "uncertainty": uncertainty,
                            "state_norm": state_norm,
                            "value": current_value,
                            "reward": reward_total,
                            "step": steps,
                            "episode": episode
                        }
                    )
                except Exception as e:
                    print("EPL evaluation failed:", e)
                    epl_enabled = False

            divergence = compute_divergence(
                state.squeeze(0),
                error_val,
                uncertainty,
                current_value,
                divergence_config
            )

            divergence += epl_soft_penalty

            barriers = compute_barriers(
                state.squeeze(0),
                error_val,
                uncertainty,
                barrier_config
            )

            for name, value in epl_barriers.items():
                barriers[f"epl:{name}"] = value

            barrier_violation = 0.0
            if barriers:
                min_barrier = min(barriers.values())
                if min_barrier < 0:
                    barrier_violation = -min_barrier

            if hasattr(env, "agentA_pos"):
                budgeter.record_visit(env.agentA_pos)
                budgeter.update(heatmap.get_average_error())
                budget_scale = budgeter.get_budget_scale(env.agentA_pos)
            else:
                budget_scale = 0.5

            gov_action = governance_controller.compute_action(
                divergence,
                barriers,
                budget_scale
            )

            if hybrid_automaton is not None:
                current_mode, transition = hybrid_automaton.step(
                    {
                        "divergence": divergence,
                        "barrier_violation": barrier_violation
                    }
                )
                if transition:
                    print("Hybrid transition:", transition)

            planner.set_budget(gov_action.horizon, gov_action.num_samples)
            planner.set_uncertainty_weight(gov_action.uncertainty_weight)

            gov_exploration_scale = gov_action.exploration_scale
            safe_mode = gov_action.safe_mode or current_mode == "safe"
            env.apply_governance(safe_mode=safe_mode, intensity=gov_action.u)

            # overlays + ordered commits (mixed consistency)
            if consistency_layer.overlay_types:
                for overlay_id, overlay_type in consistency_layer.overlay_types.items():
                    if overlay_type == "ordered":
                        consistency_layer.commit_ordered(
                            {"overlay": overlay_id, "key": "tick", "delta": 1, "step": steps}
                        )
                    elif overlay_type == "causal":
                        consistency_layer.commit_causal(
                            overlay_id,
                            {"overlay": overlay_id, "key": "tick", "delta": 1, "step": steps},
                            node_id="sim"
                        )
                    else:
                        consistency_layer.apply_overlay(overlay_id, "tick", 1)
            else:
                consistency_layer.apply_overlay("steps", "count", 1)
            if done:
                consistency_layer.commit_ordered({"event": "goal_reached", "step": steps})

            print(
                "Meta state:",
                meta_state.detach().numpy(),
                "Meta loss:",
                meta_loss,
                "Governance:",
                {
                    "u": gov_action.u,
                    "safe": gov_action.safe_mode,
                    "slack": gov_action.slack,
                    "barrier": gov_action.active_barrier,
                    "mode": current_mode
                }
            )


            states[i] = next_state

            state_trajectory.append(
                next_state.squeeze(0).detach().numpy()
            )

            total_reward += reward


        # ---------------------------------
        # Swarm Coupling
        # ---------------------------------

        alpha = 0.01

        for i in range(NUM_AGENTS):

            interaction = torch.zeros_like(states[i])

            for j in range(NUM_AGENTS):

                if i != j:
                    interaction += (states[j] - states[i])

            states[i] = states[i] + alpha * interaction


        env.render()

        steps += 1


    reward_history.append(total_reward)

    print("Episode finished")
    print("Total reward:", total_reward)


# =========================
# Plot Results
# =========================

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.plot(reward_history)
plt.title("Reward per Episode")

plt.subplot(1,3,2)
plt.plot(wm_loss_history)
plt.title("World Model Loss")

plt.subplot(1,3,3)
plt.plot(prediction_error_history)
plt.title("Prediction Error")

plt.tight_layout()
finalize_plot("training_curves")


# =========================
# Prediction Error Heatmap
# =========================

heatmap_path = os.path.join(PLOT_DIR, f"{RUN_TAG}_prediction_heatmap.png")
heatmap.plot(save_path=heatmap_path, show=not HEADLESS)
if HEADLESS:
    print("Saved plot:", heatmap_path)


# =========================
# Phase Space Plot
# =========================

state_array = np.array(state_trajectory)

plt.figure(figsize=(6,6))

if state_array.shape[1] >= 2:
    plt.plot(state_array[:,0], state_array[:,1], alpha=0.6)
else:
    plt.plot(state_array[:,0])

plt.title("Internal State Dynamics")
finalize_plot("state_dynamics")


# =========================
# World Model Dream Rollout
# =========================

print("\nRunning world model imagination rollout...")

dream_steps = 15

state = state_model.init_state()

obs = env.reset()
obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

errors = []

for t in range(dream_steps):

    action = random.randint(0,3)

    action_vec = torch.zeros(1,4)
    action_vec[0,action] = 1

    mu, logvar = world_model(state, action_vec)

    pred_state = mu

    obs_next, reward, done = env.step(action)

    obs_next = torch.tensor(
        obs_next, dtype=torch.float32
    ).unsqueeze(0)

    next_state = state_model(obs_next, action_vec, state)

    error = torch.norm(pred_state - next_state).item()

    errors.append(error)

    state = next_state


plt.figure()

plt.plot(errors)

plt.title("World Model Dream Rollout Error")

plt.xlabel("Imagination Step")
plt.ylabel("Prediction Error")

finalize_plot("dream_rollout")

import torch
import random

from environment.environment import GridWorld

from agent.state_model import StateModel
from agent.world_model import WorldModel
from agent.value_model import ValueModel
from agent.meta_model import MetaStateModel

from quantum.quantum_planner import QuantumPlanner

from training.replay_buffer import ReplayBuffer
from training.trainer import WorldModelTrainer
from training.value_trainer import ValueTrainer

from utils.heatmap import PredictionErrorHeatmap

import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# Environment
# -------------------------

env = GridWorld(size=5)

# heatmap for prediction error visualization
heatmap = PredictionErrorHeatmap(env.size)


# -------------------------
# Models
# -------------------------

state_model = StateModel()
world_model = WorldModel()
value_model = ValueModel()
meta_model = MetaStateModel()


# -------------------------
# Trainers
# -------------------------

world_trainer = WorldModelTrainer(world_model)
value_trainer = ValueTrainer(value_model)


# -------------------------
# Planner
# -------------------------

planner = QuantumPlanner(world_model, value_model)


# -------------------------
# Replay Buffer
# -------------------------

buffer = ReplayBuffer()


# -------------------------
# Training Parameters
# -------------------------

episodes = 1000
max_steps = 50


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

    # exploration decay
    epsilon = max(0.02, 0.3 * (0.990 ** episode))

    state = state_model.init_state()

    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    done = False
    steps = 0
    total_reward = 0


    while not done and steps < max_steps:

        # ---------------------------------
        # Action Selection (ε-greedy)
        # ---------------------------------

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = planner.select_action(state)

        action_vec = torch.zeros(1, 4)
        action_vec[0, action] = 1


        # ---------------------------------
        # Predict next state
        # ---------------------------------

        predicted_next_state = world_model(state, action_vec)


        # ---------------------------------
        # Environment step
        # ---------------------------------

        obs_next, reward, done = env.step(action)

        obs_next = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0)


        # ---------------------------------
        # Perturbation experiment
        # ---------------------------------

        if episode % 200 == 0 and steps == 5:
            env.goal_pos = [
                random.randint(0, env.size - 1),
                random.randint(0, env.size - 1)
            ]


        # ---------------------------------
        # Compute true next state
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

            states, actions, next_states = buffer.sample(32)

            wm_loss = world_trainer.train_step(
                states,
                actions,
                next_states
            )

            wm_loss_history.append(wm_loss)

            print("World model loss:", wm_loss)


        # ---------------------------------
        # Train Value Model
        # ---------------------------------

        reward_tensor = torch.tensor([[reward]], dtype=torch.float32)

        value_loss = value_trainer.train_step(
            state.squeeze(0).detach(),
            reward_tensor,
            next_state.squeeze(0).detach()
        )

        print("Value loss:", value_loss)


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
        # Update Heatmap
        # ---------------------------------

        if hasattr(env, "agentA_pos"):
            heatmap.update(env.agentA_pos, error_val)


        # ---------------------------------
        # Meta-State
        # ---------------------------------

        meta_state = meta_model(
            state.squeeze(0),
            error_val
        )

        print("Meta state:", meta_state.detach().numpy())

        state_trajectory.append(state.squeeze(0).detach().numpy())


        # ---------------------------------
        # Update state
        # ---------------------------------

        state = next_state
        obs = obs_next

        total_reward += reward

        env.render()

        print("Reward:", reward)
        print("Internal state norm:", torch.norm(state).item())
        print("--------------")

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
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(1,3,2)
plt.plot(wm_loss_history)
plt.title("World Model Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")

plt.subplot(1,3,3)
plt.plot(prediction_error_history)
plt.title("Prediction Error")
plt.xlabel("Step")
plt.ylabel("Error")

plt.tight_layout()
plt.show()


# =========================
# Prediction Error Heatmap
# =========================

heatmap.plot()


# =========================
# Phase Space Plot
# =========================

state_array = np.array(state_trajectory)

plt.figure(figsize=(6,6))

if state_array.shape[1] >= 2:
    plt.plot(state_array[:,0], state_array[:,1], alpha=0.6)
    plt.xlabel("State Dimension 1")
    plt.ylabel("State Dimension 2")
else:
    plt.plot(state_array[:,0])
    plt.xlabel("Time Step")
    plt.ylabel("Internal State Value")

plt.title("Internal State Dynamics")

plt.show()

# =========================
# World Model Dream Rollout
# =========================

print("\nRunning world model imagination rollout...")

dream_steps = 15

state = state_model.init_state()

obs = env.reset()
obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

real_states = []
pred_states = []

for t in range(dream_steps):

    action = random.randint(0,3)

    action_vec = torch.zeros(1,4)
    action_vec[0,action] = 1

    # predicted next state
    pred_state = world_model(state, action_vec)

    # real environment step
    obs_next, reward, done = env.step(action)
    obs_next = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0)

    next_state = state_model(obs_next, action_vec, state)

    real_states.append(next_state.detach().numpy())
    pred_states.append(pred_state.detach().numpy())

    state = next_state


# compute prediction divergence
errors = []

for r,p in zip(real_states, pred_states):

    r = torch.tensor(r)
    p = torch.tensor(p)

    errors.append(torch.norm(r-p).item())


plt.figure()

plt.plot(errors)

plt.title("World Model Dream Rollout Error")
plt.xlabel("Imagination Step")
plt.ylabel("Prediction Error")

plt.show()
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from environment.environment import GridWorld

from agent.state_model import StateModel
from agent.world_model import WorldModel
from agent.value_model import ValueModel
from agent.meta_model import MetaStateModel

from planning.mpc_planner import MPCPlanner

from training.replay_buffer import ReplayBuffer
from training.trainer import WorldModelTrainer
from training.value_trainer import ValueTrainer

from utils.heatmap import PredictionErrorHeatmap


# -------------------------
# Environment
# -------------------------

env = GridWorld(size=10)
heatmap = PredictionErrorHeatmap(env.size)


# -------------------------
# Swarm Parameters
# -------------------------

NUM_AGENTS = 4


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

planner = MPCPlanner(world_model, value_model)

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

    epsilon = max(0.02, 0.3 * (0.990 ** episode))

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

            meta_state = meta_model(
                state.squeeze(0),
                error_val,
                uncertainty
            )

            print("Meta state:", meta_state.detach().numpy())


            # ---------------------------------
            # Heatmap Update
            # ---------------------------------

            if hasattr(env, "agentA_pos"):
                heatmap.update(env.agentA_pos, error_val)


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
else:
    plt.plot(state_array[:,0])

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

plt.show()
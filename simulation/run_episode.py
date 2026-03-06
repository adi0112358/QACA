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

import matplotlib.pyplot as plt


# -------------------------
# Environment
# -------------------------

env = GridWorld(size=5)


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
        # Meta-State
        # ---------------------------------

        meta_state = meta_model(
            state.squeeze(0),
            error_val
        )

        print("Meta state:", meta_state.detach().numpy())


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
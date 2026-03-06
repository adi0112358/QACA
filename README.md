# QACA: Quantum-Assisted Cognitive Agent

QACA is an experimental cognitive agent architecture combining predictive world models, value-based planning, and quantum-assisted trajectory search.

The system demonstrates a model-based learning agent capable of predicting environment dynamics and planning actions using internal simulation.

## Mathematical Formulation

The Quantum Assisted Cognitive Agent (QACA) is a model-based cognitive architecture composed of the following components:

- latent state representation
- world model
- prediction error monitoring
- value learning
- trajectory planning
- quantum-inspired optimization

---

### Environment

The environment is modeled as a Markov Decision Process

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R)
$$

where

- $\mathcal{S}$ : state space  
- $\mathcal{A}$ : action space  
- $P(s'|s,a)$ : transition dynamics  
- $R(s,a)$ : reward function  

---

### Latent State Representation

The agent maintains an internal latent state

$$
s_t \in \mathbb{R}^d
$$

State updates follow

$$
s_t = f_\theta(o_t, a_{t-1}, s_{t-1})
$$

where

- $o_t$ : observation  
- $a_{t-1}$ : previous action  
- $f_\theta$ : state update network  

---

### World Model

The world model predicts the next latent state

$$
\hat{s}_{t+1} = g_\phi(s_t, a_t)
$$

Training minimizes prediction error

$$
\mathcal{L}_{world} =
\| s_{t+1} - \hat{s}_{t+1} \|^2
$$

---

### Prediction Error

Prediction error measures the discrepancy between predicted and actual states

$$
\epsilon_t =
\| s_{t+1} - \hat{s}_{t+1} \|
$$

This signal is used as a meta-cognitive feedback signal.

---

### Meta State

The agent computes a meta-state

$$
m_t = h_\psi(s_t, \epsilon_t)
$$

which represents internal uncertainty and model reliability.

---

### Value Function

The value function estimates expected discounted reward

$$
V(s_t) =
\mathbb{E}
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\right]
$$

Training uses temporal difference learning

$$
V(s_t) \leftarrow r_t + \gamma V(s_{t+1})
$$

Loss function

$$
\mathcal{L}_{value} =
(V(s_t) - (r_t + \gamma V(s_{t+1})))^2
$$

---

### Trajectory Planning

The planner evaluates action sequences

$$
\tau = (a_t, a_{t+1}, ..., a_{t+H})
$$

Future states are simulated using the world model

$$
s_{t+1} = g_\phi(s_t, a_t)
$$

$$
s_{t+2} = g_\phi(s_{t+1}, a_{t+1})
$$

The trajectory value is

$$
J(\tau) = V(s_{t+H})
$$

The optimal trajectory

$$
\tau^* = \arg\max_{\tau} J(\tau)
$$

---

### Quantum-Inspired Optimization

Action trajectories are encoded as bitstrings

$$
z \in \{0,1\}^{2H}
$$

Each bit pair corresponds to one action.

The cost function is

$$
C(z) = -V(s_{t+H})
$$

A QAOA-style quantum circuit generates a probability distribution

$$
p(z)
$$

The chosen trajectory is

$$
z^* = \arg\max_z p(z)
$$

The agent executes the first action of the trajectory.

## Architecture

![training curves](plots/arch_design.jpg)

## Components

environment/
GridWorld simulation used for experimentation.

agent/
Neural models representing internal state, world dynamics, value estimation, and meta-state monitoring.

quantum/
Quantum-inspired planner using a QAOA-style circuit.

training/
Training logic for world and value models.

simulation/
Training loop and evaluation.

## Features

- Predictive world modeling
- Internal state memory
- Meta-state monitoring via prediction error
- Value-based planning
- Quantum-inspired trajectory optimization
- Online training architecture

## Example Training Results

World model loss rapidly converges while prediction error decreases, demonstrating accurate learning of environment dynamics.

Example training curves:

![training curves](plots/final_results.png)

## Running the Project

Install dependencies:

pip install -r requirements.txt

Run the simulation:

python -m simulation.run_episode

## Future Work

- larger environments
- robotics simulation
- real quantum hardware backends
- multi-agent environments

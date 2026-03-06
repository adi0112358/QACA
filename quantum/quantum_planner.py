import torch
import pennylane as qml
import numpy as np


class QuantumPlanner:

    def __init__(self, world_model, value_model, depth=2):

        self.world_model = world_model
        self.value_model = value_model
        self.depth = depth

        # 2 qubits per action step
        self.n_qubits = 2 * depth

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # QAOA parameters
        self.params = np.random.randn(2)


    # --------------------------------------------------
    # Encode qubits → action
    # --------------------------------------------------

    def encode_action(self, bit1, bit2):

        if bit1 == 0 and bit2 == 0:
            return 0   # up

        if bit1 == 0 and bit2 == 1:
            return 1   # down

        if bit1 == 1 and bit2 == 0:
            return 2   # left

        if bit1 == 1 and bit2 == 1:
            return 3   # right


    # --------------------------------------------------
    # Evaluate trajectory
    # --------------------------------------------------

    def trajectory_cost(self, state, trajectory):

        current_state = state

        for action in trajectory:

            action_vec = torch.zeros(1, 4)
            action_vec[0, action] = 1

            current_state = self.world_model(current_state, action_vec)

        value = self.value_model(current_state)

        return -value.item()


    # --------------------------------------------------
    # Decode quantum bitstring → action sequence
    # --------------------------------------------------

    def decode_bitstring(self, bitstring):

        actions = []

        # ensure even length
        if len(bitstring) % 2 != 0:
            bitstring = bitstring[:-1]

        for i in range(0, len(bitstring), 2):

            a = self.encode_action(bitstring[i], bitstring[i + 1])
            actions.append(a)

        return actions


    # --------------------------------------------------
    # Build cost vector
    # --------------------------------------------------

    def build_costs(self, state):

        n_states = 2 ** self.n_qubits
        costs = []

        for i in range(n_states):

            bitstring = list(map(int, format(i, f"0{self.n_qubits}b")))

            actions = self.decode_bitstring(bitstring)

            cost = self.trajectory_cost(state, actions)

            costs.append(cost)

        return np.array(costs)


    # --------------------------------------------------
    # QAOA layer
    # --------------------------------------------------

    def qaoa_layer(self, gamma, beta):

        for i in range(self.n_qubits):
            qml.RZ(2 * gamma, wires=i)

        for i in range(self.n_qubits):
            qml.RX(2 * beta, wires=i)


    # --------------------------------------------------
    # Action selection
    # --------------------------------------------------

    def select_action(self, state):

        costs = self.build_costs(state)

        @qml.qnode(self.dev)
        def circuit(params):

            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            gamma, beta = params

            self.qaoa_layer(gamma, beta)

            return qml.probs(wires=range(self.n_qubits))

        probs = circuit(self.params)

        best_state = np.argmax(probs)

        bitstring = list(map(int, format(best_state, f"0{self.n_qubits}b")))

        trajectory = self.decode_bitstring(bitstring)

        # return first action from best trajectory
        return trajectory[0]
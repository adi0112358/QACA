import torch
import itertools


class Planner:

    def __init__(self,world_model,value_model,depth=3):

        self.world_model=world_model
        self.value_model=value_model

        self.depth=depth
        self.actions=[0,1,2,3]

    def simulate(self,state,sequence):

        current_state=state

        for action in sequence:

            action_vec=torch.zeros(1,4)
            action_vec[0,action]=1

            current_state=self.world_model(current_state,action_vec)

        return current_state


    def select_action(self,state):

        best_sequence=None
        best_value=-1e9

        for sequence in itertools.product(self.actions,repeat=self.depth):

            predicted_state=self.simulate(state,sequence)

            value=self.value_model(predicted_state).item()

            if value>best_value:

                best_value=value
                best_sequence=sequence

        return best_sequence[0]
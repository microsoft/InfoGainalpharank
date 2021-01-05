import numpy as np
import logging


class BadAgentTies:
    
    def __init__(self, actions=4, seed=None, hardness=1):
        self.logger = logging.getLogger("BadAgentTies")
        self.actions = actions
        if self.actions == 2:
            p = 1 / hardness
            self.matrix = np.array([[
                [ 0,   p],
                [ -p,  0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        if self.actions == 3:
            p = 1 / hardness
            self.matrix = np.array([[
                [ 0,   p,  1],
                [ -p,  0,  p],
                [ -1, -p,  0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        elif self.actions == 4:
            p = 1 / hardness
            self.matrix = np.array([[
                [ 0,  -p,  1, 1],
                [ p,   0,  1, 1],
                [ -1, -1,  0, 0],
                [ -1, -1,  0, 0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        elif self.actions == 5:
            p = 1 / hardness
            self.matrix = np.array([[
                [ 0, -p,  p, 1, 1],
                [ p,  0, -p, 1, 1],
                [-p,  p,  0, 1, 1],
                [ -1,  -1,  -1, 0, 0],
                [ -1,  -1,  -1, 0, 0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        elif self.actions == 6:
            p = 1 / hardness
            self.matrix = np.array([[
                [ 0, -p,  p, 1, 1, 1],
                [ p,  0, -p, 1, 1, 1],
                [-p,  p,  0, 1, 1, 1],
                [ -1,  -1,  -1, 0, 0, 0],
                [ -1,  -1,  -1, 0, 0, 0],
                [ -1,  -1,  -1, 0, 0, 0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        elif self.actions > 6:
            p = 1 / hardness
            self.matrix = np.ones(shape=(1,self.actions, self.actions))
            self.matrix[0,:3,:3] = np.array([
                [ 0, -p,  p],
                [ p,  0, -p],
                [-p,  p,  0],
            ])
            self.matrix[0,3:,:3] = -1
            self.matrix[0,:3,3:] = 1
            self.matrix[0,3:,3:] = 0
            self.matrix = (self.matrix + 1) / 2

        self.logger.debug("\n"+str(np.around(self.matrix, 2)))

    def get_entry_sample(self, entry):
        player1_win = np.random.binomial(1, self.matrix[0][tuple(entry)])
        return np.array([player1_win])

    def true_payoffs(self):
        return self.matrix
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions
import numpy as np
import gym
import torch
from torch.nn.functional import one_hot

from env.log import rpd_log, iter_log, pgg_log, pd_log


class GridSocialDilemmaEnv(gym.Env): 
    NAME = 'GridGame'
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    def __init__(self, max_steps=1, grid_size=2, k=2):
        super(GridSocialDilemmaEnv, self).__init__()
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.num_agents = 2
        self.num_actions = 2
        self.conflict_level = k

        self.initial_position_agent_0 = np.identity(self.grid_size)[0]
        self.initial_position_agent_1 = np.identity(self.grid_size)[-1]
        self.initial_state = np.concatenate([self.initial_position_agent_0, self.initial_position_agent_1],axis=0).reshape(1,-1)
        self.agent_positions = [0, self.grid_size - 1]

    def reset(self):
        self.current_step = 0
        self.agent_positions = [0, self.grid_size - 1]
        state = self.initial_state
        return state

    def step(self, actions):
        self.current_step += 1
        
        for i in range(self.num_agents):
            if actions[i]==0: # stay
                if self.agent_positions[i] > 0:
                    self.agent_positions[i] -= 1
                elif self.agent_positions[i]==0:
                    self.agent_positions[i] = 0
                else:
                    AssertionError("Negative position")
            elif actions[i]==1:
                if self.agent_positions[i] < self.grid_size - 1:
                    self.agent_positions[i] += 1
                elif self.agent_positions[i]==self.grid_size-1:
                    self.agent_positions[i] = self.grid_size - 1
                else:
                    AssertionError("Over the grid size")
        
        # reveal agents' positions
        state = np.concatenate([np.identity(self.grid_size)[self.agent_positions[0]], np.identity(self.grid_size)[self.agent_positions[1]]],axis=0).reshape(1,-1)            
        immediate_rewards = self.calculate_rewards()
        
        done = (self.current_step == self.max_steps)
        
        return state, immediate_rewards, done

    def calculate_rewards(self):
        pos_0, pos_1 = self.agent_positions
        reward_0 = pos_0 - self.conflict_level*(self.grid_size-1 - pos_1)
        reward_1 = self.grid_size-1 - pos_1 - self.conflict_level*pos_0
        rewards = np.array([reward_0, reward_1])
        return rewards

    def render(self):
        grid = [' ' for _ in range(self.grid_size)]
        if self.agent_positions[0] == self.agent_positions[1]:
            grid[self.agent_positions[0]] = 'AB'
        else:
            grid[self.agent_positions[0]] = 'A'
            grid[self.agent_positions[1]] = 'B'
        print('|'.join(grid))
        print(f"Agent A Position: {self.agent_positions[0]}, Agent B Position: {self.agent_positions[1]}")

class Dilemma:
    def __init__(self, horizon):
        self.states = [
            [[(0.0, 0.0), (2.0, -1.0)], [(-1.0, 2.0), (0.0, 0.0)]]
        ]
        self.done = 0
        self.state = 0
        self.step_count = 0
        self.dummy_state = [1., 0.]
        self.horizon = horizon
        
    def reset(self):
        self.done = 0
        self.step_count = 0
        return self.dummy_state
    
    def get_payoffs(self, action_1, action_2):
        return self.states[self.state][action_1][action_2]
        
    def step(self, action_1, action_2):
        reward = self.get_payoffs(action_1, action_2)
        self.step_count += 1
        self.done = self.step_count==self.horizon
        return self.dummy_state, reward, self.done

    def log(self, controller, rewards, pick_mediator, *args):
        info = pd_log(controller, rewards, pick_mediator)
        # info = iter_log(controller, rewards, pick_mediator)
        return info

class RandomizedPrisonersDilemma(Dilemma):
    def __init__(self):
        super().__init__()
        self.states = [
            [[(0, 0), (7, -5)], [(-5, 7), (2, 2)]],
            [[(0, 0), (1, 1)], [(1, 1), (2, 2)]]
        ]

        self.state_onehot = [[1, 0], [0, 1]]

        self.probs = [.5, .5]
        self.done = 0

    def reset(self):
        self.done = 0
        return self.change_state()

    def change_state(self):
        self.state = np.random.choice(len(self.states), p=self.probs)
        return self.state_onehot[self.state]

    def step(self, action_1, action_2):
        reward = self.get_payoffs(action_1, action_2)
        self.done = 1
        return self.state_onehot[self.state], reward, self.done

    def log(self, controller, rewards, pick_mediator, *args):
        rpd_log(controller, rewards, pick_mediator)


class IterativePrisonersDilemma(RandomizedPrisonersDilemma):
    def __init__(self):
        super().__init__()
        self.states = [
            [[(0, 0), (7, -5)], [(-5, 7), (-1, 4)]],
            [[(0, 0), (7, -5)], [(-5, 7), (2, 2)]]
        ]

        self.state_onehot = [[1, 0, 0], [0, 1, 1]]
        self.state_onehot.append([0, 0, 2])

        self.probs = [1., 0.]
        self.n_rounds = 0
        self.max_rounds = 0

    def reset(self):
        self.done = 0
        self.n_rounds = 0
        self.max_rounds = 2
        self.state = 0

        return self.state_onehot[0]
        # return self.state

    def step(self, action_1, action_2):
        reward = self.get_payoffs(action_1, action_2)
        self.state = 1

        self.n_rounds += 1
        if self.n_rounds >= self.max_rounds:
            self.done = 1
        next_state = self.state_onehot[self.n_rounds]

        return next_state, reward, self.done

    def log(self, controller, rewards, pick_mediator, *args):
        info = iter_log(controller, rewards, pick_mediator)
        # info = pd_log(controller, rewards, pick_mediator)
        return info
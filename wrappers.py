import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

class ActionRepeater(Wrapper): 
    def __init__(self, environment, repeat_count):
        super(ActionRepeater, self).__init__(environment)
        self.repeat_count = repeat_count
    
    def step(self, action):
        accumulated_reward = 0.0
        is_done = False
        for _ in range(self.repeat_count):
            next_state, reward, is_done, is_truncated, info = self.environment.step(action)
            accumulated_reward += reward
            if is_done:
                break
        return next_state, accumulated_reward, is_done, is_truncated, info
    

def enhance_environment(environment):
    environment = ActionRepeater(environment, repeat_count=4)
    environment = ResizeObservation(environment, shape=84)
    environment = GrayScaleObservation(environment)
    environment = FrameStack(environment, num_stack=4, lz4_compress=True)
    return environment

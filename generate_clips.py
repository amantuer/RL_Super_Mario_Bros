import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from agent_player import MarioAgent  #Agent

from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

import os
from PIL import Image

class FrameLogger(Wrapper):  
    def __init__(self, env, frame_skip):
        super(FrameLogger, self).__init__(env)
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.logged_frames = []
        self.logged_actions = []
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            state, reward, done, trunc, info = self.env.step(action)
            self.logged_frames.append(state.copy())
            self.logged_actions.append(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, trunc, info
    
    def reset(self, **kwargs):
        initial_state, info = self.env.reset(**kwargs)
        self.logged_frames = [initial_state.copy()]
        self.logged_actions = [0]
        return initial_state, info

def enhance_environment(env):
    env = FrameLogger(env, frame_skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env

ENVIRONMENT_NAME = 'SuperMarioBros-1-1-v0'
EPISODE_COUNT = 1000
controller_images = [Image.open(f"controllers/{i}.png") for i in range(5)]

env = gym_super_mario_bros.make(ENVIRONMENT_NAME, render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = enhance_environment(env)

mario_agent = MarioAgent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)  # Renamed agent

for episode_num in range(EPISODE_COUNT):
    done = False
    state, _ = env.reset()
    total_reward = 0

    while not done:
        action = mario_agent.select_action(state)
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        state = new_state

        if done:
            print(f"Episode: {episode_num}, Total Reward: {total_reward}")
            if info["flag_get"]:
                os.makedirs(os.path.join("games", f"game_{episode_num}"), exist_ok=True)
                frame_logger = env.env.env.env  
                for frame_index, (frame, action) in enumerate(zip(frame_logger.logged_frames, frame_logger.logged_actions)):
                    frame = Image.fromarray(frame).resize((frame.shape[1] * 10, frame.shape[0] * 10), Image.NEAREST)
                    frame.save(os.path.join("games", f"game_{episode_num}", f"frame_{frame_index}.png"))
                    controller_images[action].save(os.path.join("games", f"game_{episode_num}", f"controller_{frame_index}.png"))

env.close()

import torch
import os

import mario_env  # Renamed gym_super_mario_bros
from mario_actions import ACTION_RIGHT_ONLY  # Renamed gym_super_mario_bros.actions.RIGHT_ONLY
from agent_player import MarioAgent  # Renamed Agent
from environment_wrappers import enhance_environment  # Renamed apply_wrappers

from nes_py.wrappers import JoypadSpace
from utils import current_timestamp  # Renamed get_current_date_time_string

# Model path setup
model_directory = os.path.join("models", current_timestamp())
os.makedirs(model_directory, exist_ok=True)

# GPU Check
use_cuda = torch.cuda.is_available()
print("Using CUDA device:", torch.cuda.get_device_name(0) if use_cuda else "CUDA is not available")

# Environment settings
ENV_LEVEL = 'SuperMarioBros-1-1-v0'
TRAIN_MODE = True
DISPLAY_GAME = True
CHECKPOINT_INTERVAL = 5000
EPISODES = 50000

# Environment initialization
environment = mario_env.make(ENV_LEVEL, render_mode='human' if DISPLAY_GAME else 'rgb', apply_api_compatibility=True)
environment = JoypadSpace(environment, ACTION_RIGHT_ONLY)
environment = enhance_environment(environment)

# Agent setup
mario = MarioAgent(input_shape=environment.observation_space.shape, actions=environment.action_space.n)

# Load model for evaluation
if not TRAIN_MODE:
    model_folder = ""
    model_name = ""
    mario.load(model_directory, model_folder, model_name)
    mario.set_exploration(0.2, 0.0, 0.0)  # epsilon, eps_min, eps_decay

# Game loop
for episode in range(EPISODES):
    print(f"Starting episode: {episode}")
    done = False
    state, _ = environment.reset()
    total_rewards = 0

    while not done:
        action = mario.select_action(state)
        new_state, reward, done, truncated, info = environment.step(action)
        total_rewards += reward

        if TRAIN_MODE:
            mario.store_transition(state, action, reward, new_state, done)
            mario.update_model()

        state = new_state

    print(f"Episode: {episode}, Total Reward: {total_rewards}, Epsilon: {mario.epsilon}")

    if TRAIN_MODE and (episode + 1) % CHECKPOINT_INTERVAL == 0:
        mario.save(model_directory, f"model_ep{episode + 1}.pt")

environment.close()

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import gymnasium as gym
import numpy as np
import os
import time
from DominoEnv import DominoEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def train():
    #directory model and log saved to
    model_dir = f"models/V2/1719173400"
    log_dir = f"logs/V2/1719173400"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = DominoEnv()
    #wrap for action masking
    env = ActionMasker(env, mask_fn) 
    env.reset()
    model = MaskablePPO.load("models/V2/1719173400/490000")
    model.set_env(env)

    TIMESTEPS = 10000
    #adjust the range below to adjust timesteps it runs for (calc stepcount as max range val * timesteps)
    #saves model every TIMESTEPS number of steps
    for i in range(50,100):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        model.save(f"{model_dir}/{TIMESTEPS*i}")

def main():
    domenv = DominoEnv()
    #check environment
    check_env(domenv)

    #extra checks, unocomment to run
    # episodes = 10

    # for episode in range(episodes):
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         random_action = env.action_space.sample()
    #         print("action",random_action)
    #         obs, reward, done, trunc, info = env.step(random_action)
    #         #print("obs", obs)
    #         print('reward',reward)
    train()
    print("done")

if __name__ == "__main__":
    main()



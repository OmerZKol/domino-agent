from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from DominoEnv import DominoEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

#test agent
def main():
    test_env = DominoEnv()
    test_env = ActionMasker(test_env, mask_fn) 
    #loaded_model = MaskablePPO.load("models/Actionspace_possplays/1719170682/200000")
    #loaded_model = MaskablePPO.load("models/V2/1719158876/490000") #474, 476
    loaded_model = MaskablePPO.load("models/V2/1719173400/490000") #target_score 10: 540, 441 #target_score 61: 622, 279
    # policy = loaded_model.policy
    # mean_reward, std_reward = evaluate_policy(policy, Monitor(test_env), n_eval_episodes=10, deterministic=True)

    target_score = 10
    scores = []
    histories = []
    #make agent play 500 games against random player
    for i in range(1000):
        obs, info = test_env.reset()
        while(not test_env.unwrapped.done):
            action_masks = get_action_masks(test_env)
            action, _states = loaded_model.predict(obs, action_masks=action_masks)
            obs, reward, done, trunc, info = test_env.step(action)
        scores.append(info["score"])
        histories.append(info["history"])
    print(scores)
    overall = [0,0]
    for i in scores:
        if((i[0]>i[1]) and i[0] == target_score):
            overall[0] = overall[0]+1
        elif(i[0] < i[1] and i[1] == target_score):
            overall[1] = overall[1]+1
    print(overall)
    # for i in range(len(histories)):
        # print(i)
        # print(scores[i])
        # print(histories[i])

if __name__ == "__main__":
    main()
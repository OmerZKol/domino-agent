from stable_baselines3.common.evaluation import evaluate_policy # type: ignore
from stable_baselines3.common.monitor import Monitor # type: ignore
from DominoEnv import DominoEnv
from stable_baselines3 import PPO

#test agent
def main():
    test_env = DominoEnv()
    loaded_model = PPO.load("models/V2/1719066465/90000")
    policy = loaded_model.policy
    mean_reward, std_reward = evaluate_policy(policy, Monitor(test_env), n_eval_episodes=10, deterministic=True)

    scores = []
    #make agent play 500 games against random player
    for i in range(10):
        obs, info = test_env.reset()
        while(not test_env.done):
            action, _states = loaded_model.predict(obs)
            obs, reward, done, trunc, info = test_env.step(action)
        scores.append(info["score"])
    print(scores)
    overall = [0,0]
    for i in scores:
        if(i[0]>i[1]):
            overall[0] = overall[0]+1
        elif(i[0] < i[1]):
            overall[1] = overall[1]+1
    print(overall)

if __name__ == "__main__":
    main()
from ddpg_tf_orig import Agent
import numpy as np
import gym
import pandas as pd  # Import pandas for Series
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.compat.v1 as tf

def plotlearning(score_history, filename, window):
    
    rolling_mean = pd.Series(score_history).rolling(window).mean()
    plt.plot(rolling_mean)
    plt.title('Score Over Time')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    agent = Agent(alpha=0.001, beta=0.001, input_dims=[3], tau=0.005, env=env,
                  batch_size=64, layer1_size=400, layer2_size=300, n_actions=1)
    
    np.random.seed(0)
    score_history = []  # Initialize score_history
    best_score = -np.inf  # Initialize best_score with a very low number

    for i in range(1000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, done)
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score, '100 game average score %.2f' % avg_score)

    filename = 'pendulum.png'
    plotlearning(score_history, filename, window=100)





import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt
from ddpg_tf_orig import Agent

def plot_learning(score_history, filename, window):
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
                  batch_size=64, n_actions=1)

    np.random.seed(0)
    score_history = []
    best_score = -np.inf
    num_episodes = 1000

    for episode in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Unpack if obs is a tuple (depending on the gym version)
        done = False
        score = 0
        max_steps = 200  # Set max steps for each episode to ensure the episode ends
        steps = 0
        
        while not done and steps < max_steps:  # Ensure the episode terminates by max_steps
            act = agent.choose_action(obs)
            step_result = env.step(act)  # Get the result from the step
            
            # Print the result of each step
            print("Step result:", step_result)  
            
            # Handle different step return formats
            if len(step_result) == 4:
                new_state, reward, done, info = step_result
                truncated = False
            else:
                new_state, reward, done, truncated, info = step_result

            # Remember current experience and learn
            agent.remember(obs, act, reward, new_state, done)
            agent.learn()
            
            score += reward
            obs = new_state
            steps += 1  # Increment step count
        
        # Episode should end either when `done` is True or when max_steps is reached
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # Save models if we get a better average score
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        # Print episode summary
        print(f'Episode {episode + 1}/{num_episodes}, Score: {score:.2f}, 100 Game Avg: {avg_score:.2f}')
    
    filename = 'pendulum.png'
    plot_learning(score_history, filename, window=100)



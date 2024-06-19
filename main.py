import gym
from DQN import Agent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 500
    
    for i in range(n_games):
        score = 0
        done = False
        observation, info = env.reset(seed=42)
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            score += reward
            
            print(f"Episode {i}, Action {action}, Reward {reward}, Done {done}")

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        
        if (i+1) % 10 == 0:  # Print every 10 episodes to reduce overhead
            print(f'Episode {i+1}, Score {score:.2f}, Average Score {avg_score:.2f}, Epsilon {agent.epsilon:.2f}')
    
    x = [i + 1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plt.figure(figsize=(10, 5))
    plt.plot(x, scores, label='Score per Episode', color='blue')
    plt.plot(x, [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))], \
            label='Average Score (100 episodes)', color='red', linestyle='--')
    plt.title('Lunar Lander Training')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)

    # Plotting epsilon values
    plt.figure(figsize=(10, 5))
    plt.plot(x, eps_history, label='Epsilon', color='green')
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('epsilon_' + filename)

    plt.show()

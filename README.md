
**Overview**

This project implements a Deep Q-Network (DQN) to solve the LunarLander-v2 environment from OpenAI's Gym. The goal is for an agent to learn how to safely land a lunar module on the moon's surface using reinforcement learning techniques.

**Requirements**

- Python 3.x
- OpenAI Gym
- NumPy
- Matplotlib


**Getting Started**

**Installation**:

Clone this repository to your local machine.
Install dependencies using pip install -r requirements.txt.
Training the Agent:

Run python main.py to start training the DQN agent.
The agent's training progress will be displayed in the console, showing scores per episode, average scores over the last 100 episodes, and epsilon values (exploration rate).
Monitoring Training Progress:

During training, graphs are generated and saved:
lunar_lander.png: Plot of scores per episode and average score over the last 100 episodes.
epsilon_lunar_lander.png: Plot of epsilon values (exploration rate) over episodes.
Evaluation:

After training completes, evaluate the trained agent's performance using the saved model weights.
Fine-tune hyperparameters as needed to improve performance.
Files Description:

main.py: Main script to initialize the environment, agent, and run the training loop.
DQN.py: Contains the DQN agent implementation (Agent class).
requirements.txt: Lists the required Python packages.
Results

The trained agent learns to navigate and land the lunar module safely.
Graphs (lunar_lander.png and epsilon_lunar_lander.png) visualize the training progress, demonstrating the agent's learning curve and exploration-exploitation trade-off.

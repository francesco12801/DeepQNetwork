# Intelligent-Trading-System
Develop an intelligent trading system on EUR/USD. 

Let's start our work: 
1. The best states would be something like (asset_price, portfolio, commission). Basically, it's based on a real emulation of trading view. 
2. The actions of our agents can be: 
  - buy 
  - maintain the actual position
  - sell
3. The reward is easy; i thought about a reward based on the following criteria: 
 - Penalty for lose.
 - Reward for each gain.
4. Q-learning must be set to all zero. 

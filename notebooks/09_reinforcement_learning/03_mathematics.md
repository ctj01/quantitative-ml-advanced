# Mathematical Foundations of Reinforcement Learning

## Bellman Equations

The Bellman equation is fundamental in reinforcement learning and provides a recursive decomposition for the value function. It describes the relationship between the value of the current state and the values of subsequent states that can be reached from it.

### Bellman Equation for V(s)
For a value function V(s), the Bellman equation can be expressed as:
\[ V(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s, a) [R(s, a, s') + \gamma V(s')] \]

### Bellman Equation for Q(s, a)
For an action-value function Q(s, a), the equation is:
\[ Q(s, a) = \sum_{s' \in S} P(s'|s, a) [R(s, a, s') + \gamma \sum_{a' \in A} \pi(a'|s') Q(s', a')] \]

## Temporal Difference Learning Algorithms

### TD(0)
Temporal Difference learning estimates the value of the current state based on the estimate of the subsequent state. The update rule for TD(0) is given by:
\[ V(s) \leftarrow V(s) + \alpha [R(s, a) + \gamma V(s') - V(s)] \]

### SARSA
SARSA (State-Action-Reward-State-Action) is an on-policy TD learning algorithm. The update rule is:
\[ Q(s, a) \leftarrow Q(s, a) + \alpha [R(s, a) + \gamma Q(s', a') - Q(s, a)] \]

### Q-learning
Q-learning is an off-policy TD control algorithm that learns the value of the optimal policy regardless of the agent's actions. The update rule is:
\[ Q(s, a) \leftarrow Q(s, a) + \alpha [R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a)] \]

## Policy Gradient Theorem

The Policy Gradient Theorem provides a way to optimize the parameters of the policy directly. 
The gradient of the expected return can be expressed as:
\[ \nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) Q_w(s, a)] \]

### Formal Proof
The proof involves using the likelihood ratio and differentiating under the expectation operator. The detailed steps involve establishing the connections between policy and value functions.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

LEARNING_RATE = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
EPOCHS = 4
BATCH_SIZE = 64
ENTROPY_COEF = 0.01

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = self.fc(state)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_action_and_value(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        logits, value = self.model(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value, dist.entropy()

    def compute_advantages(self, rewards, dones, values, next_value):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * next_value * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * LAMBDA * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
        return torch.tensor(advantages, dtype=torch.float32)

    def update(self, states, actions, log_probs, returns, advantages):
        for _ in range(EPOCHS):
            indices = np.random.permutation(len(states))
            for i in range(0, len(states), BATCH_SIZE):
                idx = indices[i:i+BATCH_SIZE]
                batch_states = torch.tensor(states[idx], dtype=torch.float32)
                batch_actions = torch.tensor(actions[idx], dtype=torch.int64)
                idx = torch.tensor(idx, dtype=torch.int64)
                batch_log_probs = log_probs.index_select(0, idx)
                batch_returns = returns.index_select(0, idx)
                batch_advantages = advantages.index_select(0, idx)

                logits, values = self.model(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                print(entropy.shape, new_log_probs.shape, len(idx))

                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def train(env, agent, episodes=1000):
    for episode in range(episodes):
        state, _ = env.reset()
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        episode_reward = 0

        while True:
            action, log_prob, value, _ = agent.get_action_and_value(state)
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value.item())

            state = next_state
            episode_reward += reward

            if done:
                break

        next_value = 0 if done else agent.model(torch.tensor(state, dtype=torch.float32))[1].item()
        returns = agent.compute_advantages(rewards, dones, values, next_value) + torch.tensor(values)
        advantages = agent.compute_advantages(rewards, dones, values, next_value)

        agent.update(np.array(states), np.array(actions), torch.tensor(log_probs), returns, advantages)
        print(f"Episode {episode+1}, Reward: {episode_reward}")

if __name__ == "__main__":
    import gym
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    train(env, agent)
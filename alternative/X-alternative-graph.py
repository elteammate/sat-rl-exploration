#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch_geometric
import networkx as nx
import pyvis
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Protocol, cast
import enum
import dataclasses
import pickle
import random
import json
import uuid
import logging


# In[2]:


@dataclasses.dataclass
class Hyperparameters:
    batch_size: int = 64
    runs_per_episode: int = 64
    epochs: int = 10
    learning_rate: float = 1e-5
    eps_clip: float = 0.2
    entropy_coef: float = 0.01
    weight_decay: float = 1e-3
    value_weight: float = 0.5
    policy_weight: float = 1.0
    gae_gamma: float = 0.95
    gae_lambda: float = 0.8
    penalty_per_conflict: float = 5e-5
    temperature: float = 4.0


HP = Hyperparameters()


# In[3]:


logger = logging.getLogger("notebook")
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
np.seterr(all='raise')
DEV = "cuda"
torch.set_float32_matmul_precision("medium")

writer = SummaryWriter()

@dataclasses.dataclass
class Counters:
    episodes: int = 0 
    epochs: int = 0
    runs: int = 0
    steps: int = 0
    batches: int = 0
    train_steps: int = 0

    def from_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)

COUNTERS = Counters()


# In[4]:


class ExpectedValueNormalizationLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, ex: torch.Tensor):
        ex = torch.as_tensor(ex)

        b = torch.zeros(logits.shape[:-1], device=logits.device)

        for _ in range(100):
            normalized = torch.sigmoid(logits + b.unsqueeze(-1))
            f_gamma = normalized.sum(dim=-1) - ex
            f_prime_gamma = (normalized * (1 - normalized)).sum(dim=-1)
            diff = torch.clamp(f_gamma / f_prime_gamma, -2, 2)
            if torch.all(diff.abs() < 1e-6):
                break
            b = b - diff

        normalized = torch.sigmoid(logits + b.unsqueeze(-1))
        ctx.save_for_backward(normalized)
        return normalized

    @staticmethod
    def backward(ctx, g):
        normalized, = ctx.saved_tensors
        p_grad = normalized * (1 - normalized)
        denom = p_grad.sum(dim=-1)
        coordwise = p_grad * g

        grad = coordwise - p_grad * coordwise.sum(axis=-1).unsqueeze(-1) / denom.unsqueeze(-1)

        return grad, None


probs = torch.tensor([
    [0.999, 0.5, 0.5, 0.5, 0.1],
    [0.3, 0.5, 0.5, 0.8, 0.2],
], requires_grad=True)
x = -(1 / probs - 1).log()
y = ExpectedValueNormalizationLogits.apply(x, torch.tensor([2.0, 1.0]))
# print(x, y, y.sum(axis=-1), sep="\n")
y.sum().backward()
# print(probs.grad)

optim = torch.optim.SGD([probs], lr=0.1)
for _ in range(100):
    optim.zero_grad()
    x = -(1 / probs - 1).log()
    y = ExpectedValueNormalizationLogits.apply(x, torch.tensor([2.0, 1.0]))
    loss = y.pow(3.0).sum()
    loss.backward()
    optim.step()
    # print(probs)


# In[5]:


class GraphProblem(Protocol):
    global_data: torch.Tensor
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    reducible: torch.Tensor


# In[6]:


class ProcessingBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, disambiguate: bool):
        super().__init__()

        self.disambiguate = disambiguate

        if disambiguate:
            self.conv_reducible = gnn.GATConv(in_channels, out_channels, edge_dim=edge_dim, residual=True)
            self.conv_irreducible = gnn.GATConv(in_channels, out_channels, edge_dim=edge_dim, residual=True)
        else:
            self.conv = gnn.GATConv(in_channels, out_channels, edge_dim=edge_dim, residual=True)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, h, data):
        edge_index, edge_attr, reducible = data.edge_index, data.edge_attr, data.reducible

        out = torch.zeros_like(h)

        if self.disambiguate:
            out[reducible, :] = self.conv_reducible(h, edge_index, edge_attr)[reducible, :]
            out[~reducible, :] = self.conv_irreducible(h, edge_index, edge_attr)[~reducible, :]
        else:
            out = self.conv(h, edge_index, edge_attr)

        out = self.activation(out)

        return out


# In[7]:


class GraphModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        disambiguate_clauses_in_first: int = 2,
        edge_dim: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_fc = nn.Linear(input_dim, hidden_dims[0])
        self.silu = nn.SiLU()

        self.processing_blocks = nn.ModuleList([
            ProcessingBlock(
                hidden_dims[i],
                hidden_dims[i + 1],
                edge_dim,
                i < disambiguate_clauses_in_first
            )
            for i in range(len(hidden_dims) - 1)
        ])

        self.output_fc = nn.Linear(hidden_dims[-1], 1, bias=False)
        self.value_fc_reducible = nn.Linear(hidden_dims[-1], 1, bias=False)
        self.value_fc_irreducible = nn.Linear(hidden_dims[-1], 1)

    def forward(self, graph: GraphProblem, ex: torch.Tensor | float):
        logger.info("Starting model forward pass")

        input_ = torch.cat([
            graph.global_data.view((1, -1)).expand(graph.x.shape[0], -1),
            graph.x,
        ], dim=-1)

        x = self.silu(self.input_fc(input_))

        for block in self.processing_blocks:
            x = block(x, graph)

        logits = self.output_fc(x[graph.reducible]).view(-1)

        probs = ExpectedValueNormalizationLogits.apply(logits * HP.temperature, ex)
        if torch.isnan(probs).any():
            print(logits)
            print(probs)
            raise AssertionError()

        value_reducible = self.value_fc_reducible(x[graph.reducible]).view(-1)
        value_irreducible = self.value_fc_irreducible(x[~graph.reducible]).view(-1)
        value = torch.cat([value_reducible, value_irreducible]).sum()

        logger.info("Finished model forward pass")

        return probs, value


# In[8]:


def compute_returns_advantages(rewards: list[float], values: list[float]) -> tuple[list[float], list[float]]:
    returns = []
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + HP.gae_gamma * (values[i + 1] if i + 1 < len(values) else 0) - values[i]
        gae = delta + HP.gae_gamma * HP.gae_lambda * gae
        returns.append(gae + values[i])
        advantages.append(gae)
    return returns[::-1], advantages[::-1]


# In[9]:


@dataclasses.dataclass
class EpisodeResult:
    states: list[GraphProblem]
    dists: list[torch.distributions.Bernoulli]
    actions: list[torch.Tensor]
    rewards: list[float]
    values: list[float]
    returns: list[float]
    advantages: list[float]

    stats: list[dict]

    @staticmethod
    def empty() -> 'EpisodeResult':
        return EpisodeResult([], [], [], [], [], [], [], [])

    def merge_with(self, other: 'EpisodeResult') -> 'EpisodeResult':
        assert (
            len(self.states) == len(self.dists) == len(self.actions) ==
            len(self.values) == len(self.rewards) == len(self.returns) ==
            len(self.advantages)
        )
        assert (
            len(other.states) == len(other.dists) == len(other.actions) ==
            len(other.values) == len(other.rewards) == len(other.returns) ==
            len(other.advantages)
        )
        return EpisodeResult(
            self.states + other.states,
            self.dists + other.dists,
            self.actions + other.actions,
            self.rewards + other.rewards,
            self.values + other.values,
            self.returns + other.returns,
            self.advantages + other.advantages,
            self.stats + other.stats,
        )

    @staticmethod
    def merge_all(results: list['EpisodeResult']) -> 'EpisodeResult':
        result = results[0]
        for other in results[1:]:
            result = result.merge_with(other)
        return result

    def add(self, *, state, dist, action, reward, value):
        assert (
            len(self.states) == len(self.dists) == len(self.actions) ==
            len(self.values) == len(self.rewards)
        )
        self.states.append(state)
        self.dists.append(dist)
        self.actions.append(action)
        if reward is not None:
            self.rewards.append(reward)
        self.values.append(value)

    def add_reward(self, reward):
        assert (
            len(self.states) == len(self.dists) == len(self.actions) ==
            len(self.values) == len(self.rewards) + 1
        )
        self.rewards.append(reward)

    def complete(self, stats: dict):
        assert (
            len(self.states) == len(self.dists) == len(self.actions) ==
            len(self.values) == len(self.rewards)
        )
        assert len(self.stats) == 0
        self.returns, self.advantages = compute_returns_advantages(self.rewards, self.values)
        self.stats = [stats]


# In[10]:


class Agent:
    def __init__(self, model):
        self.model = model.to(DEV)
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=HP.learning_rate,
            weight_decay=HP.weight_decay
        )

    def act(self, graph: GraphProblem, ex: float):
        COUNTERS.steps += 1
        logger.info("Running agent act")
        with torch.no_grad():
            graph = graph.to(DEV)
            assert torch.isnan(graph.x).sum() == 0
            assert torch.isnan(graph.edge_attr).sum() == 0
            assert torch.isnan(graph.global_data).sum() == 0
            probs, value = self.model(graph, ex)
            logger.info("Finished agent act")
            return torch.distributions.Bernoulli(probs), value.item()

    def update(self, results: EpisodeResult, silent: bool = False):
        logger.info("Training agent")
        r = results

        n = len(r.states)
        batch_size = HP.batch_size
        advantages = torch.tensor(r.advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for num_epoch in range(HP.epochs):
            COUNTERS.epochs += 1
            logger.info("Training agent, epoch %d", num_epoch)
            # TODO: resamples based on advantages
            indices = random.sample(range(n), n)
            for batch_start in range(0, n, batch_size):
                COUNTERS.batches += 1
                idx = indices[batch_start:batch_start + batch_size]

                self.optim.zero_grad()
                for i in idx:
                    COUNTERS.train_steps += 1
                    graph, ex = r.states[i]
                    new_probs, value = self.model(graph, ex)
                    value = value.cpu()
                    dist = torch.distributions.Bernoulli(probs=new_probs)
                    log_probs = dist.log_prob(r.actions[i].float())
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(log_probs - r.dists[i].log_prob(r.actions[i].float()))
                    surr1 = ratio * advantages[i]
                    surr2 = torch.clamp(ratio, 1.0 - HP.eps_clip, 1.0 + HP.eps_clip) * advantages[i]

                    value_loss = torch.nn.functional.huber_loss(value, torch.tensor(r.returns[i], dtype=torch.float32))
                    policy_loss = -torch.min(surr1, surr2).mean()

                    if not silent:
                        writer.add_scalar("value loss", value_loss.item(), COUNTERS.train_steps)

                    loss = (
                        HP.policy_weight * policy_loss
                        + HP.value_weight * value_loss
                        - HP.entropy_coef * entropy
                    ) / len(idx)

                    loss.backward()

                self.optim.step()


# In[11]:


@dataclasses.dataclass
class GameConfig:
    number_of_bits: int = 4
    number_of_cards_to_add: tuple[int] = (16, 8, 12, 16, 24, 32, 48, 48, 48, 48)
    fraction_to_remove: float = 0.5
    fraction_to_make_reducible: float = 0.2


# In[12]:


class ToyEnv:
    def __init__(self, game_cfg: GameConfig, runs_per_episode: int = 16):
        self.runs_per_episode = runs_per_episode
        self.game_cfg = game_cfg

    def run_instance(self, agent: Agent):
        COUNTERS.runs += 1
        result = EpisodeResult.empty()

        max_card = 2 ** self.game_cfg.number_of_bits
        deck = np.random.randint(0, max_card, size=self.game_cfg.number_of_cards_to_add[0])
        reducible = np.ones_like(deck, dtype=bool)
        total_reward = 0

        for step in range(len(self.game_cfg.number_of_cards_to_add) - 1):
            COUNTERS.steps += 1
            logger.info("Running instance step %d", step)

            problem = Data(
                global_data=torch.tensor([
                    len(deck) / sum(self.game_cfg.number_of_cards_to_add),
                ], dtype=torch.float32),
                x=torch.tensor(np.array([
                    deck / (max_card - 1),
                    *[np.bitwise_and(deck, 1 << i) >> i for i in range(self.game_cfg.number_of_bits)],
                ]), dtype=torch.float32).permute(1, 0),
                edge_index=torch.tensor(np.array([
                    np.arange(len(deck)),
                    np.arange(1, len(deck) + 1) % len(deck),
                ]), dtype=torch.long),
                edge_attr=torch.zeros((len(deck), 0), dtype=torch.float32),
                reducible=torch.tensor(reducible, dtype=torch.bool),
            )

            ex = self.game_cfg.fraction_to_remove * sum(reducible)
            dist, value = agent.act(problem, ex)
            action = dist.sample()

            indices = np.arange(len(deck))[reducible][action.cpu().numpy() == 1]
            deck = np.delete(deck, indices)
            reducible = np.delete(reducible, indices)

            reducible = np.random.rand(*reducible.shape) < self.game_cfg.fraction_to_make_reducible

            reward = deck.sum() + np.bitwise_xor(deck[1:], deck[:-1]).sum() + (np.bitwise_xor(deck[-1], deck[0]) if len(deck) > 1 else 0)
            reward /= max_card * sum(self.game_cfg.number_of_cards_to_add)
            total_reward += reward

            result.add(state=(problem, ex), dist=dist, action=action, reward=reward, value=value)

            deck = list(deck)
            reducible = list(reducible)

            for _ in range(self.game_cfg.number_of_cards_to_add[step + 1]):
                card = random.randrange(max_card)
                index = random.randrange(len(deck) + 1) 
                deck.insert(index, card)
                reducible.insert(index, True)

            deck = np.array(deck)
            reducible = np.array(reducible)

            logger.info("Finished instance step %d", step)

        result.complete({"total_reward": total_reward})
        return result

    def run_episode(self, agent: Agent) -> EpisodeResult:
        results = []
        for _ in range(self.runs_per_episode):
            results.append(self.run_instance(agent))

        return EpisodeResult.merge_all(results)


# In[13]:


logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.WARNING)

game_cfg = GameConfig()

model = GraphModel(
    input_dim=game_cfg.number_of_bits + 2,
    edge_dim=0,
    disambiguate_clauses_in_first=1,
    hidden_dims=[32, 32, 32]
)

agent = Agent(model)

env = ToyEnv(
    game_cfg=game_cfg,
    runs_per_episode=HP.runs_per_episode,
)


# In[ ]:


CHECKPOINT_ROOT = Path("alternative-checkpoints-graph")
CHECKPOINT_ROOT.mkdir(exist_ok=True, parents=True)
saved_checkpoints = list(CHECKPOINT_ROOT.glob("*.pt"))
saved_checkpoints.sort(key=lambda p: p.stat().st_mtime)

if len(saved_checkpoints) > 0:
    checkpoint = torch.load(saved_checkpoints[-1])
    agent.model.load_state_dict(checkpoint["model"])
    agent.optim.load_state_dict(checkpoint["optim"])
    COUNTERS.from_dict(checkpoint["counters"])


# In[15]:


while True:
    print(f"Episode {COUNTERS.episodes}")
    writer.add_scalar("episode", COUNTERS.episodes, COUNTERS.episodes)
    results = env.run_episode(agent)
    logger.info("Finished episode, starting training")
    agent.update(results)

    torch.save({
        "model": agent.model.state_dict(),
        "optim": agent.optim.state_dict(),
        "counters": dataclasses.asdict(COUNTERS),
    }, CHECKPOINT_ROOT / "agent-{COUNTERS.episodes}-{uuid.uuid4()}.pt")

    writer.add_scalar("episode_reward", sum(results.rewards), COUNTERS.episodes)
    print(f"Rewards: {sum(results.rewards)}")
    COUNTERS.episodes += 1
    del results


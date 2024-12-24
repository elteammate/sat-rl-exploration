#%%
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

import srunner

from pathlib import Path
from typing import Protocol, cast
import enum
import dataclasses
import pickle
import random
import json
import uuid
import logging
#%%
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
#%%
from primitives import Clause

class NodeType(enum.Enum):
    VARIABLE = 0
    REDUNDANT = 1
    IRREDUNDANT = 2

class CnfGraph(Protocol):
    global_data: torch.Tensor
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_type: torch.Tensor

@dataclasses.dataclass
class DimInfo:
    num_global_features: int
    num_var_features: int
    num_clause_features: int
    num_edge_features: int

@dataclasses.dataclass
class ReductionProblem:
    num_vars: int
    levels: list[int]
    vals: list[int]
    clauses: list[Clause]
    reducible_ids: list[int]
    conflicts: int
#%%
def problem_to_cnf_graph(problem: ReductionProblem) -> tuple[CnfGraph, DimInfo]:
    logger.info("Converting problem to CNF graph")
    num_vars = problem.num_vars

    fixed_vals = [problem.vals[i] if problem.levels[i] == 0 else 0 for i in range(num_vars)]

    clauses = []
    for clause in problem.clauses:
        new_lits = []
        for lit in clause.lits:
            v = abs(lit) - 1
            if fixed_vals[v] == 0:
                new_lits.append(lit)
            elif lit > 0 and fixed_vals[v] == 1 or lit < 0 and fixed_vals[v] == -1:
                break
            # else skip literal because it's falsified
        else:
            clauses.append(clause.with_lits(new_lits))

    num_clauses = len(clauses)
    num_free_vars = sum(1 for v in fixed_vals if v == 0)

    c_clause_sizes = np.array([len(clause.lits) for clause in clauses])
    c_clause_lbd = np.array([clause.lbd for clause in clauses])
    g_max_clause_size = c_clause_sizes.max()
    g_min_clause_size = c_clause_sizes.min()
    g_mean_clause_size = c_clause_sizes.mean()
    g_min_clause_lbd = c_clause_lbd.min()
    g_max_clause_lbd = c_clause_lbd.max()
    g_lbd_spread = g_max_clause_lbd - g_min_clause_lbd
    g_mean_clause_lbd = c_clause_lbd.mean()
    c_lbd_size_ratio = c_clause_lbd / c_clause_sizes
    c_num_pos_literals = np.array([sum(1 for lit in clause.lits if lit > 0) for clause in clauses])
    c_num_neg_literals = c_clause_sizes - c_num_pos_literals
    c_pos_neg_ratios = c_num_pos_literals / c_clause_sizes
    g_max_pos_neg_ratio = c_pos_neg_ratios.max()
    g_min_pos_neg_ratio = c_pos_neg_ratios.min()
    g_mean_pos_neg_ratio = c_pos_neg_ratios.mean()
    c_horn = c_num_pos_literals <= 1
    g_horn_ratio = c_horn.mean()
    c_inv_horn = c_num_neg_literals <= 1
    g_inv_horn_ratio = c_inv_horn.mean()
    g_size_spread = g_max_clause_size - g_min_clause_size
    g_pos_neg_ratio_spread = g_max_pos_neg_ratio - g_min_pos_neg_ratio
    c_conflicts_on_creation = np.array([clause.conflicts_on_creation for clause in clauses])
    c_time = c_conflicts_on_creation / problem.conflicts
    c_activity = np.array([clause.activity for clause in clauses], dtype=np.float32)
    c_activity /= max(c_activity.max(), 1.0)
    c_activity_top10 = c_activity > np.percentile(c_activity, 90)
    c_time_top10 = c_time > np.percentile(c_time, 90)
    c_times_reason = np.array([clause.times_reason for clause in clauses], dtype=np.float32)
    c_times_reason /= max(c_times_reason.max(), 1.0)
    c_times_reason_top10 = c_times_reason > np.percentile(c_times_reason, 90)


    v_pos = np.zeros(num_vars)
    v_neg = np.zeros(num_vars)
    v_horn = np.zeros(num_vars)
    v_invhorn = np.zeros(num_vars)

    for i, clause in enumerate(clauses):
        for lit in clause.lits:
            v = abs(lit) - 1
            if lit > 0:
                v_pos[v] += 1
            else:
                v_neg[v] += 1
            if c_horn[i]:
                v_horn[v] += 1
            if c_inv_horn[i]:
                v_invhorn[v] += 1

    v_count = v_pos + v_neg
    v_count_non_zero = np.maximum(v_count, 1)
    v_pos_neg_ratios = v_pos / v_count_non_zero
    v_horn_ratio = v_horn / v_count_non_zero
    v_invhorn_ratio = v_invhorn / v_count_non_zero
    g_var_horn_min = v_horn.min()
    g_var_horn_max = v_horn.max()
    g_var_horn_mean = v_horn.mean()
    g_var_invhorn_min = v_invhorn.min()
    g_var_invhorn_max = v_invhorn.max()
    g_var_invhorn_mean = v_invhorn.mean()
    g_var_horn_std = v_horn.std()
    g_var_invhorn_std = v_invhorn.std()
    g_var_pos_neg_ratio_std = v_pos_neg_ratios.std()
    g_var_count_std = v_count.std()
    g_var_clause_ratio = num_free_vars / num_clauses
    g_free_ratio = num_free_vars / num_vars

    c_clause_keep = np.array([clause.keep for clause in clauses])
    c_clause_redundant = np.array([clause.redundant for clause in clauses])

    global_features = [
        g_mean_clause_size,
        g_mean_clause_lbd,
        g_mean_pos_neg_ratio,
        g_horn_ratio,
        g_inv_horn_ratio,
        g_var_horn_mean,
        g_var_invhorn_mean,
        g_var_horn_std,
        g_var_invhorn_std,
        g_var_pos_neg_ratio_std,
        g_var_count_std,
        g_var_clause_ratio,
        g_free_ratio,
    ]

    clause_features = [
        c_clause_sizes,
        c_clause_sizes / num_vars,
        (c_clause_sizes - g_min_clause_size) / g_size_spread if g_size_spread > 0 else 0,
        c_clause_lbd,
        (c_clause_lbd - g_min_clause_lbd) / g_lbd_spread if g_lbd_spread > 0 else 0,
        c_lbd_size_ratio,
        c_pos_neg_ratios,
        c_horn,
        c_inv_horn,
        c_clause_redundant,
        c_clause_keep,
        c_time,
        c_time_top10,
        c_activity,
        c_activity_top10,
        c_times_reason,
        c_times_reason_top10,
    ]

    var_features = [
        v_pos,
        v_neg,
        v_pos_neg_ratios,
        v_horn,
        v_invhorn,
        v_horn_ratio,
        v_invhorn_ratio,
    ]

    embedding_size = max(len(var_features), len(clause_features))

    global_data = torch.tensor(global_features, dtype=torch.float32)
    x = torch.zeros(num_vars + num_clauses, embedding_size)
    edge_index = []
    edge_attr = []
    node_type = torch.zeros(num_vars + num_clauses, dtype=torch.int8)

    node_type[:num_vars] = NodeType.VARIABLE.value
    node_type[num_vars:] = NodeType.IRREDUNDANT.value

    reducible_ids = set(problem.reducible_ids)

    for i, clause in enumerate(clauses):
        if clause.id_ in reducible_ids:
            node_type[num_vars + i] = NodeType.REDUNDANT.value

    for i, f in enumerate(var_features):
        x[:num_vars, i] = torch.tensor(f, dtype=torch.float32)

    for i, f in enumerate(clause_features):
        x[num_vars:, i] = torch.tensor(f, dtype=torch.float32)

    for i, clause in enumerate(clauses):
        for lit in clause.lits:
            v = abs(lit) - 1
            edge_index.append((v, num_vars + i))
            edge_attr.append(-1 if lit < 0 else 1)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1)

    logger.info("Finished converting problem to CNF graph")

    return cast(CnfGraph, Data(
        global_data=global_data,
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.int64).t(),
        edge_attr=edge_attr,
        node_type=node_type,
    )), DimInfo(
        num_clause_features=len(clause_features),
        num_edge_features=edge_attr.size(dim=1),
        num_global_features=len(global_features),
        num_var_features=len(var_features),
    )


def ordered_reducible_ids(problem: ReductionProblem) -> list[int]:
    reducible_ids = set(problem.reducible_ids)
    return [clause.id_ for clause in problem.clauses if clause.id_ in reducible_ids]
#%%
minimal_example_data, DIM_INFO = problem_to_cnf_graph(ReductionProblem(
    2,
    [1, 1],
    [0, 0],
    [
        Clause(0, [1, 2], 0, False, True, False),
        Clause(5, [-1, -2], 0, False, False, False),
        Clause(8, [-1, 2], 0, False, False, False),
    ],
    [5, 8],
    3,
))

print(DIM_INFO)
#%%
with open("../archives/example-problem.pkl", "rb") as f:
    problem = pickle.load(f)

example_data, _ = problem_to_cnf_graph(problem)

nx_graph = torch_geometric.utils.to_networkx(example_data, to_undirected=True)

g = pyvis.network.Network(width=1800, height=1000, cdn_resources='in_line')
g.toggle_hide_edges_on_drag = True
g.barnes_hut()
g.from_nx(nx_graph)
g.save_graph("example.html")

eccentricities = list(nx.eccentricity(nx_graph).values())
plt.bar(*np.unique(eccentricities, return_counts=True))
#%%
class CnfProcessingBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, disambiguate_clauses: bool):
        super().__init__()

        self.disambiguate_clauses = disambiguate_clauses

        if disambiguate_clauses:
            self.conv_redundant = gnn.GATConv(in_channels, out_channels, edge_dim=edge_dim, residual=True)
            self.conv_irredundant = gnn.GATConv(in_channels, out_channels, edge_dim=edge_dim, residual=True)
        else:
            self.conv_clauses = gnn.GATConv(in_channels, out_channels, edge_dim=edge_dim, residual=True)

        self.conv_variables = gnn.GATConv(in_channels, out_channels, edge_dim=edge_dim, residual=True)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, h, data):
        edge_index, edge_attr, node_type = data.edge_index, data.edge_attr, data.node_type

        out = torch.zeros_like(h)

        variable_mask = (node_type == NodeType.VARIABLE.value)
        out[variable_mask, :] = self.conv_variables(h, edge_index, edge_attr)[variable_mask, :]

        if self.disambiguate_clauses:
            redundant_mask = (node_type == NodeType.REDUNDANT.value)
            irredundant_mask = (node_type == NodeType.IRREDUNDANT.value)

            out[redundant_mask, :] = self.conv_redundant(h, edge_index, edge_attr)[redundant_mask, :]
            out[irredundant_mask, :] = self.conv_irredundant(h, edge_index, edge_attr)[irredundant_mask, :]
        else:
            clause_mask = (node_type != NodeType.VARIABLE.value)

            out[clause_mask] = self.conv_clauses(h, edge_index, edge_attr)[clause_mask, :]


        out = self.activation(out)

        return out
#%%
class ExpectedValueNormalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p: torch.Tensor, ex: torch.Tensor):
        ex = torch.as_tensor(ex)

        gamma = torch.ones(p.shape[:-1], device=p.device)

        for _ in range(10):
            normalized = p ** gamma.unsqueeze(-1)
            f_gamma = normalized.sum(dim=-1) - ex
            f_prime_gamma = (normalized * p.log()).sum(dim=-1)
            new_gamma = gamma - f_gamma / f_prime_gamma
            gamma = torch.maximum(new_gamma, gamma / 10)

        normalized = p ** gamma.unsqueeze(-1)
        ctx.save_for_backward(p, normalized, gamma)
        return normalized

    @staticmethod
    def backward(ctx, grad_output):
        p, normalized, gamma = ctx.saved_tensors

        normalized_dp = normalized * p.log()
        denom = normalized_dp.sum(dim=-1)

        jac = (
            torch.diag_embed(gamma.unsqueeze(-1) * normalized / p) -
            (gamma / denom).unsqueeze(-1).unsqueeze(-1) * normalized_dp.unsqueeze(-2) * (normalized / p).unsqueeze(-1)
        )

        return (jac @ grad_output.unsqueeze(-1)).squeeze(-1), None


"""
x = torch.tensor([
    [1., 0.5, 0.5, 0.5, 0.1],
    [0.3, 0.5, 0.5, 0.8, 0.2],
], requires_grad=True)
optim = torch.optim.SGD([x], lr=0.1)
for _ in range(100):
    optim.zero_grad()
    y = ExpectedValueNormalization.apply(x, torch.tensor([2.0, 1.0]))
    loss = y.pow(3.0).sum()
    loss.backward()
    optim.step()
    print(x.sum(axis=1))
"""
#%%
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
#%%
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

#%%
class CnfGraphModel(nn.Module):
    def __init__(
        self,
        input_global_dim: int,
        input_variable_dim: int,
        input_clause_dim: int,
        hidden_dims: list[int],
        disambiguate_clauses_in_first: int = 2,
        edge_dim: int = 1,
    ):
        super().__init__()

        self.input_variable_dim = input_variable_dim
        self.input_clause_dim = input_clause_dim

        self.variable_fc = nn.Linear(input_global_dim + input_variable_dim, hidden_dims[0])
        self.clause_fc = nn.Linear(input_global_dim + input_clause_dim, hidden_dims[0])

        self.silu = nn.SiLU()

        self.processing_blocks = nn.ModuleList([
            CnfProcessingBlock(
                hidden_dims[i],
                hidden_dims[i + 1],
                edge_dim,
                i < disambiguate_clauses_in_first
            )
            for i in range(len(hidden_dims) - 1)
        ])

        self.output_fc = nn.Linear(hidden_dims[-1], 1, bias=False)

        self.value_fc_clauses = nn.Linear(hidden_dims[-1], 1, bias=False)
        self.value_fc_variables = nn.Linear(hidden_dims[-1], 1)

    def forward(self, cnf_graph: CnfGraph, ex: torch.Tensor | float):
        logger.info("Starting model forward pass")

        num_vertices = cnf_graph.x.shape[0]
        variable_mask = (cnf_graph.node_type == NodeType.VARIABLE.value)
        num_variables = variable_mask.sum()

        variable_input = torch.cat([
            cnf_graph.global_data.expand(num_variables, -1),
            cnf_graph.x[:num_variables, :self.input_variable_dim],
        ], dim=-1)

        variable_embedding = self.silu(self.variable_fc(variable_input))

        clause_input = torch.cat([
            cnf_graph.global_data.expand(num_vertices - num_variables, -1),
            cnf_graph.x[num_variables:, :self.input_clause_dim],
        ], dim=-1)

        clause_embedding = self.silu(self.clause_fc(clause_input))

        x = torch.cat([variable_embedding, clause_embedding], dim=0)

        for block in self.processing_blocks:
            x = block(x, cnf_graph)

        logits = self.output_fc(x[cnf_graph.node_type == NodeType.REDUNDANT.value]).view(-1)
        # logits = (logits - logits.mean()) / logits.std()
        # assert not torch.isnan(logits).any()
        # denorm_probs = torch.sigmoid(logits)
        # assert not torch.isnan(denorm_probs).any()
        # probs = ExpectedValueNormalization.apply(denorm_probs, ex)
        # if torch.isnan(probs).any():
        #     print(denorm_probs)
        #     print(probs)
        #     raise AssertionError()
        probs = ExpectedValueNormalizationLogits.apply(logits * HP.temperature, ex)
        if torch.isnan(probs).any():
            print(logits)
            print(probs)
            raise AssertionError()

        value_vars = self.value_fc_variables(x[variable_mask])
        value_clauses = self.value_fc_clauses(x[~variable_mask])
        value = torch.cat([value_vars, value_clauses]).mean()

        logger.info("Finished model forward pass")

        return probs, value
#%%
testing_model = CnfGraphModel(
    input_global_dim=DIM_INFO.num_global_features,
    input_variable_dim=DIM_INFO.num_var_features,
    input_clause_dim=DIM_INFO.num_clause_features,
    edge_dim=DIM_INFO.num_edge_features,
    hidden_dims=[32, 32, 32, 32],
)

testing_model(minimal_example_data, 0.75)
#%%
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
#%%
@dataclasses.dataclass
class EpisodeResult:
    states: list[CnfGraph]
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

#%%
class Agent:
    def __init__(self, model):
        self.model = model.to(DEV)
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=HP.learning_rate,
            weight_decay=HP.weight_decay
        )

    def act(self, cnf_graph: CnfGraph, ex: float):
        COUNTERS.steps += 1
        logger.info("Running agent act")
        with torch.no_grad():
            cnf_graph = cnf_graph.to(DEV)
            assert torch.isnan(cnf_graph.x).sum() == 0
            assert torch.isnan(cnf_graph.edge_attr).sum() == 0
            assert torch.isnan(cnf_graph.global_data).sum() == 0
            probs, value = self.model(cnf_graph, ex)
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

                    value_loss = torch.nn.functional.huber_loss(value, torch.tensor(r.returns[i]))
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
#%%
testing_agent = Agent(testing_model)

states = [
    (minimal_example_data, 0.75),
    (minimal_example_data, 0.5),
    (minimal_example_data, 0.25),
]

dist1, value1 = testing_agent.act(*states[0])
dist2, value2 = testing_agent.act(*states[1])
dist3, value3 = testing_agent.act(*states[2])

dists = [dist1, dist2, dist3]
values = [value1, value2, value3]
print(values)

actions = [dist.sample() > 0.5 for dist in dists]
rewards = [-1.0, -1.0, 10.0]

returns, advantages = compute_returns_advantages(rewards, [1.0, 1.0, 1.0])
print(advantages)

results = EpisodeResult(
    states,
    dists,
    actions,
    rewards,
    values,
    returns,
    advantages,
    [],
)

testing_agent.update(results, silent=True)

#%%
@dataclasses.dataclass
class CnfProblemInstance:
    result: bool
    path: Path
    stats: dict[str, any]
    stats_no_reductions: dict[str, any]


with open("../archives/runs.json", "r") as f:
    runs = [json.loads(line) for line in f.readlines()]
    runs = [CnfProblemInstance(
        result=run["result"],
        path=Path(run["path"]),
        stats=run["stats"],
        stats_no_reductions=run["stats_no_reductions"],
    ) for run in runs]


def get_random_instance(unsat_only: bool = True) -> CnfProblemInstance:
    while True:
        run = random.choice(runs)
        if not unsat_only or not run.result:
            return run
#%%
class CadicalEnv:
    def __init__(
        self,
        cadical_path: Path,
        runs_per_episode: int = 16,
    ):
        self.cadical_path = cadical_path
        self.runs_per_episode = runs_per_episode

    def run_instance(self, agent: Agent, instance: CnfProblemInstance) -> EpisodeResult:
        COUNTERS.runs += 1
        result = EpisodeResult.empty()
        router = srunner.Router()

        reduce_number = 0
        last_conflicts = 0

        @router.route("reduce")
        def _reduce_route(conn: srunner.Connection, info: srunner.RunInfo, _data):
            nonlocal reduce_number
            nonlocal last_conflicts

            logger.info("Received clause database reduction")

            num_vars = conn.read_u64()
            levels = [-1] * num_vars
            vals = [-1] * num_vars
            for i in range(num_vars):
                vals[i] = conn.read_i8()
                levels[i] = conn.read_i32()
            num_clauses = conn.read_u64()
            clauses = [conn.read_clause() for _ in range(num_clauses)]
            num_reducible = conn.read_u64()
            num_target = conn.read_u64()
            reducible_ids = [conn.read_u64() for _ in range(num_reducible)]
            conflicts = conn.read_u64()
            conn.write_ok()

            if reduce_number != 0:
                result.add_reward(-(conflicts - last_conflicts) * HP.penalty_per_conflict)
            last_conflicts = conflicts

            problem = ReductionProblem(
                num_vars=num_vars,
                levels=levels,
                vals=vals,
                clauses=clauses,
                reducible_ids=reducible_ids,
                conflicts=conflicts,
            )

            logger.debug("Finished reading reduction data")

            cnf, dim_info = problem_to_cnf_graph(problem)
            assert dim_info == DIM_INFO

            ex = float(num_target)

            logger.debug("Converted problem to CNF graph")

            dist, value = agent.act(cnf, ex)
            action = dist.sample()

            logger.debug("Agent acted")

            result.add(
                state=(cnf, ex),
                dist=dist,
                action=action,
                reward=None,
                value=value,
            )

            logger.debug("Results are written to replay buffer")

            reducible_ids = ordered_reducible_ids(problem)

            to_reduces = []
            for id_, a in zip(reducible_ids, action):
                if a > 0.5:
                    to_reduces.append(id_)

            logger.debug("Sending reduction data")

            conn.write_u32(len(to_reduces))
            for id_ in to_reduces:
                conn.write_u64(id_)

            reduce_number += 1

        @router.route("stats")
        def _stats_route(conn: srunner.Connection, info: srunner.RunInfo, data):
            nonlocal last_conflicts

            logger.info("Received stats")
            stats = {}
            while (name := conn.read_str()) != "end":
                if name == "time":
                    stats[name] = conn.read_f64()
                else:
                    stats[name] = conn.read_u64()
            conn.write_ok()

            for name, value in stats.items():
                writer.add_scalar(f"Runs/{name}", value, COUNTERS.runs)
                if instance.stats[name]:
                    writer.add_scalar(f"Runs relative/{name}", value / instance.stats[name], COUNTERS.runs)

            writer.add_scalar(f"run_conflicts", stats["conflicts"], COUNTERS.runs)
            writer.add_scalar(f"relative_lbd_conflicts", stats["conflicts"] / instance.stats["conflicts"], COUNTERS.runs)
            writer.add_scalar(f"relative_random_conflicts", stats["conflicts"] / instance.stats_no_reductions["conflicts"], COUNTERS.runs)

            result.add_reward(-(stats["conflicts"] - last_conflicts) * HP.penalty_per_conflict)
            result.complete(stats)

        srunner.run_instance(
            self.cadical_path,
            ["--reduce-mode", "2"],
            instance.path,
            router.routes,
            silent=True,
        )

        return result

    def run_episode(self, agent: Agent) -> EpisodeResult:
        results = []
        for _ in range(self.runs_per_episode):
            instance = get_random_instance()
            results.append(self.run_instance(agent, instance))

        return EpisodeResult.merge_all(results)

#%%
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.WARNING)

model = CnfGraphModel(
    input_global_dim=DIM_INFO.num_global_features,
    input_variable_dim=DIM_INFO.num_var_features,
    input_clause_dim=DIM_INFO.num_clause_features,
    edge_dim=DIM_INFO.num_edge_features,
    hidden_dims=[32, 32, 32, 32],
)

agent = Agent(model)

env = CadicalEnv(
    Path("../cadical/build/cadical"),
    runs_per_episode=HP.runs_per_episode,
)
#%%
CHECKPOINT_ROOT = Path("../checkpoints")
CHECKPOINT_ROOT.mkdir(exist_ok=True, parents=True)
saved_checkpoints = list(CHECKPOINT_ROOT.glob("*.pt"))
saved_checkpoints.sort(key=lambda p: p.stat().st_mtime)

if len(saved_checkpoints) > 0:
    checkpoint = torch.load(saved_checkpoints[-1])
    agent.model.load_state_dict(checkpoint["model"])
    agent.optim.load_state_dict(checkpoint["optim"])
    COUNTERS.from_dict(checkpoint["counters"])
#%%
# torch.cuda.memory._record_memory_history(max_entries=10 ** 7)
#%%
while True:
    COUNTERS.episodes += 1
    print(f"Episode {COUNTERS.episodes}")
    writer.add_scalar("episode", COUNTERS.episodes, COUNTERS.episodes)
    results = env.run_episode(agent)
    logger.info("Finished episode, starting training")
    agent.update(results)

    torch.save({
        "model": agent.model.state_dict(),
        "optim": agent.optim.state_dict(),
        "counters": dataclasses.asdict(COUNTERS),
    }, f"../checkpoints/agent-{COUNTERS.episodes}-{uuid.uuid4()}.pt")

    writer.add_scalar("episode_reward", sum(results.rewards), COUNTERS.episodes)
    print(f"Rewards: {sum(results.rewards)}")
    del results

#%%
torch.cuda.memory._dump_snapshot(f"memory.pickle")

# Stop recording memory snapshot history.
torch.cuda.memory._record_memory_history(enabled=None)
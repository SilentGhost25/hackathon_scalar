"""
DQN Agent for Stock Trading
============================

Deep Q-Network implementation in PyTorch with:
  • Replay buffer
  • Target network (periodic hard sync)
  • Epsilon-greedy exploration with decay
  • Model save / load utilities
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DQN_CONFIG


# ============================================================
# Q-Network
# ============================================================

class QNetwork(nn.Module):
    """
    Simple feed-forward Q-network.

    Architecture:
        Input(state_size) → 128 → LayerNorm → GELU → 128 → LayerNorm → GELU → Output(action_size)
    """

    def __init__(self, state_size: int, action_size: int, hidden1: int = 128, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Linear(hidden2, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """Fixed-size experience replay buffer with uniform sampling."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:
    """
    DQN agent with target network and epsilon-greedy exploration.

    Parameters
    ----------
    state_size : int
        Dimensionality of the observation vector.
    action_size : int
        Number of discrete actions.
    config : dict, optional
        Override DQN_CONFIG values.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Optional[dict] = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.cfg = {**DQN_CONFIG, **(config or {})}

        # Device (CPU — hackathon requirement: must run without GPU)
        self.device = torch.device("cpu")

        # Networks
        h1 = self.cfg["hidden_dim_1"]
        h2 = self.cfg["hidden_dim_2"]
        self.policy_net = QNetwork(state_size, action_size, h1, h2).to(self.device)
        self.target_net = QNetwork(state_size, action_size, h1, h2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly

        # Optimiser
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.cfg["learning_rate"]
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        # Replay buffer
        self.memory = ReplayBuffer(self.cfg["replay_buffer_size"])

        # Exploration
        self.epsilon = self.cfg["epsilon_start"]
        self.epsilon_end = self.cfg["epsilon_end"]
        self.epsilon_decay = self.cfg["epsilon_decay"]

        # Training state
        self.gamma = self.cfg["gamma"]
        self.batch_size = self.cfg["batch_size"]
        self.train_step_count = 0

    # ----------------------------------------------------------------
    # Action selection
    # ----------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        evaluate : bool
            If True, always exploit (no exploration).

        Returns
        -------
        action : int
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax(dim=1).item()

    # ----------------------------------------------------------------
    # Learning
    # ----------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch from replay buffer and perform one
        gradient step on the policy network.

        Returns
        -------
        loss : float or None
            Training loss, or None if buffer is too small.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Target Q-values (from frozen target network)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Compute loss and back-propagate
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1

        return loss.item()

    def update_target_network(self) -> None:
        """Hard-copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Save model weights to disk."""
        path = path or self.cfg["save_path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step_count": self.train_step_count,
            },
            path,
        )
        return path

    def load(self, path: Optional[str] = None) -> None:
        """Load model weights from disk."""
        path = path or self.cfg["save_path"]
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.train_step_count = checkpoint.get("train_step_count", 0)

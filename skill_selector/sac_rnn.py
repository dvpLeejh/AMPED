import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class SkillActor(nn.Module):

    # A recurrent skill actor that uses GRU for temporal modeling
    def __init__(self, obs_dim: int, skill_dim: int, feature_dim: int,
                 hidden_dim: int, rnn_hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.policy = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, skill_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle both single observation and sequence cases
        single = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            single = True
        batch, seq_len, obs_dim = obs.shape
        obs_flat = obs.reshape(batch * seq_len, obs_dim)
        features = self.trunk(obs_flat)
        features = features.reshape(batch, seq_len, -1)
        if hidden is None:
            h0 = torch.zeros(1, batch, self.rnn.hidden_size, device=obs.device)
            hidden = h0
        gru_out, new_hidden = self.rnn(features, hidden)
        last_hidden = gru_out[:, -1, :]
        logits = self.policy(last_hidden)
        return logits if single else (logits, new_hidden)


class SkillCritic(nn.Module):
    """
    Critic network for skill selection. Given an observation, it outputs Q-values
    for each skill. The architecture mirrors that of the original MLP-based
    critic from the SAC baseline to ensure a fair comparison. It consists of a
    small trunk followed by a two-layer MLP.
    """

    def __init__(self, obs_dim: int, skill_dim: int, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # Feature extractor: linear layer, layer norm and Tanh activation
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        # Q-network: two-layer MLP producing a Q-value for each skill
        self.q_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, skill_dim),
        )
        # Initialize weights to match SAC baseline
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of observations."""
        h = self.trunk(obs)
        q_values = self.q_net(h)
        return q_values
        
class SkillSelectorAgent:
    """
        A skill selector agent that uses SAC with RNN for skill selection
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        skill_dim: int,
        device: torch.device,
        actor_lr: float,
        critic_lr: float,
        feature_dim: int,
        hidden_dim: int,
        update_every_steps: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 20000,
        rnn_hidden_dim: int = 128,
        max_trajectory_len: int = 100,
    ) -> None:
        self.skill_dim = skill_dim
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_every_steps = update_every_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rnn_hidden_dim = rnn_hidden_dim
        self.max_trajectory_len = max_trajectory_len
        self.step_count = 0

        self.encoder = nn.Identity()
        self.obs_dim = obs_shape[0]

        self.actor = SkillActor(
            self.obs_dim, skill_dim, feature_dim, hidden_dim, rnn_hidden_dim
        ).to(device)
        self.critic = SkillCritic(
            self.obs_dim, skill_dim, feature_dim, hidden_dim
        ).to(device)
        self.target_critic = SkillCritic(
            self.obs_dim, skill_dim, feature_dim, hidden_dim
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.encoder_opt = None

        # buffer to accumulate observations within an episode. Used only at inference
        # time to feed the full trajectory to the RNN when acting. This is not used
        # for policy updates, which are done using replay samples for stability.
        self.trajectory_buffer: List[np.ndarray] = []
        self.hidden_state: Optional[torch.Tensor] = None

    # Reset the trajectory buffer and hidden state for a new episode
    def reset_episode(self) -> None:
        self.trajectory_buffer = []
        self.hidden_state = None

    def get_epsilon(self, step: int) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -float(step) / self.epsilon_decay
        )
    def train(self, training: bool = True) -> None:
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_opt is not None:
            self.encoder.train(training)

    def eval(self) -> None:
        self.train(False)

    def act(self, obs: np.ndarray, step: int, eval_mode: bool = False) -> np.ndarray:
        # record the observation and construct the full trajectory tensor. the full
        # trajectory up to the current step is used to infer the skill distribution
        self.step_count = step
        self.trajectory_buffer.append(obs)
        if self.max_trajectory_len > 0 and len(self.trajectory_buffer) > self.max_trajectory_len:
            self.trajectory_buffer.pop(0)
        trajectory = np.array(self.trajectory_buffer, dtype=np.float32)
        traj_tensor = torch.as_tensor(trajectory, device=self.device).unsqueeze(0)
        encoded_traj = self.encoder(traj_tensor)
        # feed the full trajectory into the recurrent actor; hidden_state is updated internally
        with torch.no_grad():
            logits, self.hidden_state = self.actor(encoded_traj, self.hidden_state)
        dist = torch.distributions.Categorical(logits=logits)
        epsilon = self.get_epsilon(step)
        if eval_mode:
            skill_idx = torch.argmax(logits, dim=-1)
        else:
            if random.random() < epsilon:
                skill_idx = torch.tensor([random.randrange(self.skill_dim)], device=self.device)
            else:
                skill_idx = dist.sample()
        skill_onehot = np.zeros(self.skill_dim, dtype=np.float32)
        skill_onehot[skill_idx.item()] = 1.0
        return skill_onehot

    def update(
        self,
        replay_iter,
        step: int,
        gamma: float = 0.99,
        tau: float = 0.005,
    ) -> dict:
        
        """
        Update the critic and actor networks. The update consists of two parts:
            
            First, the critic is updated using samples from the replay buffer in an
            off-policy Q-learning manner.
            
            Second, the actor is updated by feeding the accumulated observation sequence
            from the current episode into the GRU to select skills, and then using the
            Q-values for the last observation to update the policy.
            
            Args:
                replay_iter: Iterator providing batches from the replay buffer.
                step: Current global step (used for logging and epsilon decay).
                gamma: Discount factor for Q-learning targets.
                tau: Soft update coefficient for target networks.
            
            Returns:
                metrics: Dictionary containing critic and actor loss values.
        """
        
        metrics: dict = {}

        # --- critic update ---
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, *meta = utils.to_torch(batch, self.device)
        stored_skill = meta[0] if len(meta) > 0 else None
        obs_enc = self.encoder(obs)
        next_obs_enc = self.encoder(next_obs)

        q_values = self.critic(obs_enc)
        if stored_skill is not None:
            skill_idx = stored_skill.argmax(dim=1, keepdim=True)
        else:
            logits = self.actor(obs_enc)
            dist = torch.distributions.Categorical(logits=logits)
            sampled_skill = dist.sample()
            skill_idx = sampled_skill.unsqueeze(1)
        current_q = q_values.gather(1, skill_idx)

        with torch.no_grad():
            next_q = self.target_critic(next_obs_enc)
            next_q_max, _ = next_q.max(dim=1, keepdim=True)
            target_q = reward + discount * gamma * next_q_max

        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        metrics["skill_critic_loss"] = critic_loss.item()

        # --- actor update (sequence-based) ---
        # When training the actor, use the current episode's trajectory as a
        # sequence to provide temporal context to the GRU. If there is no trajectory yet (e.g.
        # at the very beginning), skip the update to avoid computing gradients
        # on an empty buffer.
        if len(self.trajectory_buffer) > 0:
            # Build a (1, seq_len, obs_dim) tensor of the current trajectory
            traj = np.array(self.trajectory_buffer, dtype=np.float32)
            traj_tensor = torch.as_tensor(traj, device=self.device).unsqueeze(0)
            encoded_traj = self.encoder(traj_tensor)
            # Forward pass through the GRU over the entire sequence; ignore the
            # returned hidden state because we only need the final logits
            logits_seq, _ = self.actor(encoded_traj, None)
            dist = torch.distributions.Categorical(logits=logits_seq)
            sampled_skill = dist.sample()  # (batch=1,)
            log_prob = dist.log_prob(sampled_skill)
            # Evaluate the Q-value for the last observation in the trajectory
            last_obs = traj_tensor[:, -1, :]
            last_obs_enc = self.encoder(last_obs)
            q_vals_actor = self.critic(last_obs_enc)
            actor_q = q_vals_actor.gather(1, sampled_skill.unsqueeze(1)).squeeze(1)
            entropy = dist.entropy()
            beta = 0.01
            actor_loss = -(log_prob * actor_q) - beta * entropy
            actor_loss = actor_loss.mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_opt.step()
            metrics["skill_actor_loss"] = actor_loss.item()
        else:
            # No trajectory yet; skip actor update
            metrics["skill_actor_loss"] = 0.0

        utils.soft_update_params(self.critic, self.target_critic, tau)
        return metrics

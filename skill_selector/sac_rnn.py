import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import utils


class Memory(nn.Module):
    """GRU-based memory module for processing sequences"""
    def __init__(self, obs_dim, skill_dim, device, hidden_size=256):
        super().__init__()
        self.skill_dim = skill_dim
        self.device = device
        self.hidden_size = hidden_size
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # GRU: input = encoded_obs + skill + reward
        self.gru = nn.GRU(
            input_size=hidden_size + skill_dim + 1,  # +1 for reward
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.apply(utils.weight_init)
    
    def forward(self, obs, prev_skill, prev_rew, hidden, training=False):
        # Encode observation
        obs_enc = self.obs_encoder(obs)
        
        # Concatenate inputs
        gru_input = torch.cat([obs_enc, prev_skill, prev_rew], dim=-1)
        
        # Handle batch vs single step
        if training:
            # During training: (batch, seq_len, features)
            if gru_input.dim() == 2:
                gru_input = gru_input.unsqueeze(0)
            gru_out, hidden_out = self.gru(gru_input, hidden)
            if gru_out.dim() == 3 and gru_out.size(0) == 1:
                gru_out = gru_out.squeeze(0)
        else:
            # During inference: (1, 1, features)
            if gru_input.dim() == 1:
                gru_input = gru_input.unsqueeze(0).unsqueeze(1)
            elif gru_input.dim() == 2:
                gru_input = gru_input.unsqueeze(1)
            gru_out, hidden_out = self.gru(gru_input, hidden)
            gru_out = gru_out.squeeze(1)
        
        return gru_out, hidden_out


class SkillPolicy(nn.Module):
    """Categorical policy for skill selection"""
    def __init__(self, hidden_size, skill_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, skill_dim)
        )
        self.apply(utils.weight_init)
    
    def forward(self, memory_emb):
        return self.net(memory_emb)
    
    def sample(self, memory_emb):
        logits = self.forward(memory_emb)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Entropy for exploration
        entropy = dist.entropy()
        
        return action, probs, log_prob, entropy


class SkillQFunction(nn.Module):
    """Q-network for skill values"""
    def __init__(self, obs_dim, skill_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, skill_dim)
        )
        self.apply(utils.weight_init)
    
    def forward(self, obs):
        return self.net(obs)


class SACRNNSkillSelector:
    def __init__(
        self,
        obs_shape,
        skill_dim,
        device,
        lr,
        hidden_dim,
        update_every_steps,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        **kwargs
    ):
        self.skill_dim = skill_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.update_every_steps = update_every_steps
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        
        self.encoder = nn.Identity()
        self.obs_dim = obs_shape[0]
        
        # Networks
        self.memory = Memory(self.obs_dim, skill_dim, device, hidden_dim).to(device)
        self.policy = SkillPolicy(hidden_dim, skill_dim).to(device)
        
        # Twin Q-functions
        self.q1 = SkillQFunction(self.obs_dim, skill_dim, hidden_dim).to(device)
        self.q2 = SkillQFunction(self.obs_dim, skill_dim, hidden_dim).to(device)
        self.q1_target = SkillQFunction(self.obs_dim, skill_dim, hidden_dim).to(device)
        self.q2_target = SkillQFunction(self.obs_dim, skill_dim, hidden_dim).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            list(self.memory.parameters()) + list(self.policy.parameters()), 
            lr=lr
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), 
            lr=lr
        )
        
        # Hidden state for current episode
        self.hidden = None
        self._reset_hidden()
        
        # # Previous action and reward
        # self.prev_skill = None
        # self.prev_reward = 0.0
    
    
    def _reset_hidden(self):
        """Reset hidden state at episode boundaries"""
        self.hidden = torch.zeros(1, 1, self.hidden_dim).to(self.device)
        self.prev_skill = torch.zeros(self.skill_dim).to(self.device)
        self.prev_reward = 0.0
    
    def train(self, training=True):
        self.memory.train(training)
        self.policy.train(training)
        self.q1.train(training)
        self.q2.train(training)
    
    def eval(self):
        self.train(False)
    
    def act(self, obs, step, eval_mode=False):
        obs_tensor = torch.as_tensor(obs, device=self.device).float()
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # Get memory embedding
        prev_reward_tensor = torch.tensor([[self.prev_reward]], device=self.device).float()
        
        with torch.no_grad():
            memory_emb, self.hidden = self.memory(
                obs_tensor, 
                self.prev_skill.unsqueeze(0), 
                prev_reward_tensor, 
                self.hidden,
                training=False
            )
            
            if eval_mode:
                # Greedy action
                logits = self.policy(memory_emb)
                skill_idx = torch.argmax(logits, dim=-1).item()
            else:
                # Sample action
                skill_idx, _, _, _ = self.policy.sample(memory_emb)
                skill_idx = skill_idx.item()
        
        # Convert to one-hot
        skill_onehot = np.zeros(self.skill_dim).astype(np.float32)
        skill_onehot[skill_idx] = 1.0
        
        # Store for next step
        self.prev_skill = torch.tensor(skill_onehot, device=self.device)
        
        return skill_onehot
    
    def update_episode_info(self, reward):
        """Update reward info after environment step"""
        self.prev_reward = reward
    
    def reset_episode(self):
        """Reset at episode boundaries"""
        self._reset_hidden()
    
    def update(self, replay_buffer, step):
        """Trajectory-based update for RNN"""
        metrics = {}
        
        if step % self.update_every_steps != 0:
            return metrics
        
        # Check if buffer has episodes
        if not hasattr(replay_buffer, '_episode_fns') or len(replay_buffer._episode_fns) == 0:
            return metrics
        
        batch_size = 8  # Number of trajectories
        total_q_loss = 0
        total_policy_loss = 0
        total_entropy = 0
        valid_updates = 0
        
        for _ in range(batch_size):
            # Sample trajectory
            traj = replay_buffer.sample_trajectory(seq_length=20)
            if traj is None:
                continue
            
            actual_len = traj['actual_length']
            if actual_len < 2:  # Need at least 2 steps
                continue
            
            # Convert to tensors
            obs_seq = torch.tensor(traj['observation'], device=self.device, dtype=torch.float32)
            reward_seq = torch.tensor(traj['reward'], device=self.device, dtype=torch.float32)
            discount_seq = torch.tensor(traj['discount'], device=self.device, dtype=torch.float32)
            
            # Get skills if available
            if 'skill' in traj:
                skill_seq = torch.tensor(traj['skill'], device=self.device, dtype=torch.float32)
            else:
                # Use dummy skills
                skill_seq = torch.zeros(actual_len, self.skill_dim, device=self.device)
            
            # Reset hidden state for episode
            hidden = torch.zeros(1, 1, self.hidden_dim).to(self.device)
            
            # Forward pass through trajectory
            q_losses = []
            policy_losses = []
            entropies = []
            
            for t in range(actual_len - 1):
                # Current step
                obs = obs_seq[t].unsqueeze(0)
                skill = skill_seq[t].unsqueeze(0)
                reward = reward_seq[t].unsqueeze(0).unsqueeze(0)
                next_obs = obs_seq[t + 1].unsqueeze(0)
                discount = discount_seq[t].unsqueeze(0).unsqueeze(0)
                
                # Get memory embedding
                memory_emb, hidden_new = self.memory(obs, skill, reward, hidden, training=True)
                
                # Q-values for current state
                q1_values = self.q1(obs)
                q2_values = self.q2(obs)
                skill_idx = skill.argmax(dim=1, keepdim=True)
                q1 = q1_values.gather(1, skill_idx)
                q2 = q2_values.gather(1, skill_idx)
                
                # Target Q-value
                with torch.no_grad():
                    # Next memory embedding
                    next_memory_emb, _ = self.memory(
                        next_obs, skill, reward, hidden_new, training=True
                    )
                    
                    # Sample next action
                    _, next_probs, next_log_prob, _ = self.policy.sample(next_memory_emb)
                    
                    # Target Q-values
                    q1_target_values = self.q1_target(next_obs)
                    q2_target_values = self.q2_target(next_obs)
                    q_target_values = torch.min(q1_target_values, q2_target_values)
                    
                    # Soft Q-value
                    v_target = (next_probs * (q_target_values - self.alpha * next_log_prob.unsqueeze(1))).sum(dim=1, keepdim=True)
                    target_q = reward + discount * self.gamma * v_target
                
                # Q-losses
                q1_loss = F.mse_loss(q1, target_q)
                q2_loss = F.mse_loss(q2, target_q)
                q_losses.append(q1_loss + q2_loss)
                
                # Policy loss
                _, probs, log_prob, entropy = self.policy.sample(memory_emb)
                
                with torch.no_grad():
                    q_policy = torch.min(q1_values, q2_values)
                
                expected_q = (probs * q_policy).sum(dim=1)
                policy_loss = -(expected_q + self.alpha * entropy).mean()
                
                policy_losses.append(policy_loss)
                entropies.append(entropy.mean())
                
                # Update hidden state
                hidden = hidden_new.detach()
            
            if q_losses:
                # Average losses over trajectory
                avg_q_loss = torch.stack(q_losses).mean()
                avg_policy_loss = torch.stack(policy_losses).mean()
                avg_entropy = torch.stack(entropies).mean()
                
                # Update Q-functions
                self.q_optimizer.zero_grad()
                avg_q_loss.backward(retain_graph=True)
                self.q_optimizer.step()
                
                # Update policy
                self.policy_optimizer.zero_grad()
                avg_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.memory.parameters()) + list(self.policy.parameters()), 
                    max_norm=10.0
                )
                self.policy_optimizer.step()
                
                total_q_loss += avg_q_loss.item()
                total_policy_loss += avg_policy_loss.item()
                total_entropy += avg_entropy.item()
                valid_updates += 1
        
        if valid_updates > 0:
            # Update target networks
            utils.soft_update_params(self.q1, self.q1_target, self.tau)
            utils.soft_update_params(self.q2, self.q2_target, self.tau)
            
            metrics['sac_rnn_q_loss'] = total_q_loss / valid_updates
            metrics['sac_rnn_policy_loss'] = total_policy_loss / valid_updates
            metrics['sac_rnn_entropy'] = total_entropy / valid_updates
        
        return metrics
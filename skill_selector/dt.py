import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import transformers

import utils
from replay_buffer import episode_len

class SkillDecisionTransformer(nn.Module):
    """
    Decision Transformer adapted for skill selection.
    Uses GPT2 to model (Return_1, obs_1, skill_1, Return_2, obs_2, skill_2, ...)
    """
    
    def __init__(
        self,
        obs_dim,
        skill_dim,
        hidden_size,
        n_layer=3,
        n_head=4,
        max_length=20,
        max_ep_len=1000,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        config = transformers.GPT2Config(
            vocab_size=1,  # not used
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=hidden_size * 4,
            activation_function='gelu',
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            embd_pdrop=dropout,
            n_positions=1024,
        )
        
        # Custom GPT2 without positional embeddings
        self.transformer = transformers.GPT2Model(config)
        
        # Embedding layers
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_obs = nn.Linear(obs_dim, hidden_size)
        self.embed_skill = nn.Linear(skill_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # Prediction heads
        self.predict_skill = nn.Linear(hidden_size, skill_dim)
        self.predict_return = nn.Linear(hidden_size, 1)
        
        self.apply(utils.weight_init)
    
    def forward(self, obs, skills, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = obs.shape[0], obs.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=obs.device)
        
        # Embed each modality
        obs_embeddings = self.embed_obs(obs)
        skill_embeddings = self.embed_skill(skills)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        
        # Add time embeddings
        obs_embeddings = obs_embeddings + time_embeddings
        skill_embeddings = skill_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # Stack as (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, obs_embeddings, skill_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Stack attention mask
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        # Forward through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs.last_hidden_state
        
        # Reshape back
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        
        # Get predictions
        # Predict next skill given obs (and implicitly the return-to-go)
        skill_preds = self.predict_skill(x[:, 1])  # from obs embeddings
        return_preds = self.predict_return(x[:, 2])  # from skill embeddings
        
        return skill_preds, return_preds


class DTSkillSelector:
    def __init__(
        self,
        obs_shape,
        skill_dim,
        device,
        lr,
        weight_decay,
        warmup_steps,
        context_length,
        n_layer,
        n_head,
        hidden_dim,
        dropout,
        target_return,
        update_every_steps,
        **kwargs
    ):
        self.skill_dim = skill_dim
        self.device = device
        self.context_length = context_length
        self.target_return = target_return
        self.update_every_steps = update_every_steps
        
        self.encoder = nn.Identity()
        self.obs_dim = obs_shape[0]
        
        # Initialize Decision Transformer
        self.model = SkillDecisionTransformer(
            obs_dim=self.obs_dim,
            skill_dim=skill_dim,
            hidden_size=hidden_dim,
            n_layer=n_layer,
            n_head=n_head,
            max_length=context_length,
            dropout=dropout,
        ).to(device)
        
        # Optimizer with warmup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.warmup_steps = warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
        
        # Episode storage for context
        self.current_trajectory = {
            'observations': deque(maxlen=context_length),
            'skills': deque(maxlen=context_length),
            'rewards': deque(maxlen=context_length),
            'timesteps': deque(maxlen=context_length),
        }
        self.episode_return = 0
        self.episode_length = 0
    
    def train(self, training=True):
        self.model.train(training)
    
    def eval(self):
        self.train(False)
    
    def act(self, obs, step, eval_mode=False):
        with torch.no_grad():
            # Add current observation to trajectory
            self.current_trajectory['observations'].append(obs)
            self.current_trajectory['timesteps'].append(self.episode_length)
            
            # Convert trajectory to tensors
            if len(self.current_trajectory['observations']) == 0:
                # First step - return random skill
                skill_idx = np.random.randint(self.skill_dim)
                skill_onehot = np.zeros(self.skill_dim).astype(np.float32)
                skill_onehot[skill_idx] = 1.0
                self.current_trajectory['skills'].append(skill_onehot)
                return skill_onehot
            
            # Prepare inputs
            obs_seq = torch.tensor(
                np.array(list(self.current_trajectory['observations'])),
                dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            
            # For first action, use dummy previous skills
            if len(self.current_trajectory['skills']) == 0:
                skill_seq = torch.zeros((1, 1, self.skill_dim), device=self.device)
            else:
                skill_seq = torch.tensor(
                    np.array(list(self.current_trajectory['skills'])),
                    dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            
            # Calculate returns-to-go
            rewards = list(self.current_trajectory['rewards'])
            returns_to_go = []
            if len(rewards) > 0:
                cumsum = np.cumsum(rewards[::-1])[::-1]
                target_return = self.target_return if eval_mode else self.target_return
                for i in range(len(rewards)):
                    returns_to_go.append(target_return - cumsum[i])
            returns_to_go.append(self.target_return)  # For current step
            
            returns_to_go = torch.tensor(
                returns_to_go[-obs_seq.shape[1]:],
                dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(-1)
            
            timesteps = torch.tensor(
                list(self.current_trajectory['timesteps'])[-obs_seq.shape[1]:],
                dtype=torch.long, device=self.device
            ).unsqueeze(0)
            
            # Pad sequences if needed
            if skill_seq.shape[1] < obs_seq.shape[1]:
                # Pad skills with zeros at the beginning
                pad_len = obs_seq.shape[1] - skill_seq.shape[1]
                skill_seq = torch.cat([
                    torch.zeros((1, pad_len, self.skill_dim), device=self.device),
                    skill_seq
                ], dim=1)
            
            # Get skill prediction
            skill_logits, _ = self.model(
                obs_seq, skill_seq, returns_to_go, timesteps
            )
            
            # Get the last predicted skill
            logits = skill_logits[0, -1]
            
            if eval_mode:
                skill_idx = torch.argmax(logits)
            else:
                # Sample from distribution
                dist = torch.distributions.Categorical(logits=logits)
                skill_idx = dist.sample()
            
            skill_onehot = np.zeros(self.skill_dim).astype(np.float32)
            skill_onehot[skill_idx] = 1.0
            
            # Store the selected skill (will be paired with next observation)
            self.current_trajectory['skills'].append(skill_onehot)
            
            return skill_onehot
    
    def update_trajectory(self, reward):
        """Call this after environment step to update trajectory with reward"""
        self.current_trajectory['rewards'].append(reward)
        self.episode_return += reward
        self.episode_length += 1
    
    def reset_trajectory(self):
        """Reset trajectory storage at episode boundaries"""
        self.current_trajectory = {
            'observations': deque(maxlen=self.context_length),
            'skills': deque(maxlen=self.context_length),
            'rewards': deque(maxlen=self.context_length),
            'timesteps': deque(maxlen=self.context_length),
        }
        self.episode_return = 0
        self.episode_length = 0
    
    def update(self, replay_buffer, step):
        """Updated method that works with AMPED's replay buffer structure"""
        metrics = {}
        
        if step % self.update_every_steps != 0:
            return metrics
        
        if not hasattr(replay_buffer, '_episode_fns') or len(replay_buffer._episode_fns) == 0:
            return metrics
                    
        batch_size = 32
        
        # Direct trajectory sampling from buffer
        trajectories = []
        for _ in range(batch_size):
            # Sample episode and create trajectory
            episode = replay_buffer._sample_episode()
            ep_len = episode_len(episode)
            
            if ep_len <= self.context_length:
                start_idx = 1
                actual_length = ep_len
            else:
                start_idx = np.random.randint(1, ep_len - self.context_length + 2)
                actual_length = self.context_length
            
            end_idx = start_idx + actual_length
            
            traj = {
                'observation': episode['observation'][start_idx-1:end_idx-1],
                'reward': episode['reward'][start_idx:end_idx],
                'actual_length': actual_length,
            }
            
            # Add skill if available
            if 'skill' in episode:
                traj['skill'] = episode['skill'][start_idx-1:end_idx-1]
            
            trajectories.append(traj)
        
        # Prepare batch tensors
        max_len = self.context_length
        obs_seq = torch.zeros((batch_size, max_len, self.obs_dim), device=self.device)
        skill_seq = torch.zeros((batch_size, max_len, self.skill_dim), device=self.device)
        returns_to_go = torch.zeros((batch_size, max_len, 1), device=self.device)
        timestep_seq = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch_size, max_len), device=self.device)
        
        for i, traj in enumerate(trajectories):
            actual_len = traj['actual_length']
            
            # Fill tensors
            obs_seq[i, :actual_len] = torch.tensor(traj['observation'], device=self.device)
            if 'skill' in traj:
                skill_seq[i, :actual_len] = torch.tensor(traj['skill'], device=self.device)
            
            # Calculate returns-to-go
            rewards = traj['reward']
            for t in range(actual_len):
                returns_to_go[i, t, 0] = self.target_return - np.sum(rewards[t:])
            
            # Timesteps
            timestep_seq[i, :actual_len] = torch.arange(actual_len, device=self.device)
            attention_mask[i, :actual_len] = 1.0
        
        # Forward pass
        skill_preds, return_preds = self.model(
            obs_seq, skill_seq, returns_to_go, timestep_seq, attention_mask
        )
        
        # Compute losses
        mask = attention_mask.flatten()
        skill_targets = skill_seq.argmax(dim=-1).flatten()
        skill_preds_flat = skill_preds.view(-1, self.skill_dim)
        
        # Only compute loss on valid positions
        valid_mask = mask == 1
        if valid_mask.sum() > 0:
            skill_loss = F.cross_entropy(
                skill_preds_flat[valid_mask],
                skill_targets[valid_mask]
            )
            
            return_loss = F.mse_loss(
                return_preds.flatten()[valid_mask],
                returns_to_go.flatten()[valid_mask]
            )
        else:
            skill_loss = torch.tensor(0.0, device=self.device)
            return_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = skill_loss + 0.1 * return_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        self.scheduler.step()
        
        metrics['dt_skill_loss'] = skill_loss.item()
        metrics['dt_return_loss'] = return_loss.item()
        metrics['dt_total_loss'] = total_loss.item()
        metrics['dt_lr'] = self.scheduler.get_last_lr()[0]
        
        return metrics
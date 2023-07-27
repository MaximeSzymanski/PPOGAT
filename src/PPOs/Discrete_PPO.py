from typing import List
import numpy as np
import torch
import dataclasses
from tqdm import tqdm
import time
from src.model.Discrete.LSTM.LSTMCritic import LSTMCritic
from src.model.Discrete.LSTM.LSTMActor import LSTMActor
from src.model.Discrete.MLP.MLPActor import MLPActor
from src.model.Discrete.MLP.MLPCritic import MLPCritic
from src.PPOs.AbstractPPO import AbstractPPO
from src.model.Discrete.GAT.Actor import GATActor
from src.model.Discrete.GAT.Critic import GATCritic
from src.gym_env.GridWorld import GridWorld_copsVSthief
def get_model_flattened_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


@dataclasses.dataclass
class DiscretePPO(AbstractPPO):
    """Discrete Proximal Policy Optimization (PPO) agent."""

    def __post_init__(self) -> None:
        """Perform post initialization checks and setup any additional attributes"""
        self.continuous_action_space = False
        super().__post_init__()

        self.action_size = 5
        self.env = GridWorld_copsVSthief(10,10)
        self.state_size = 10
        self.recurrent = False
        self.actor = GATActor(
            nfeat=1, nhid=16, noutput=self.action_size, dropout=0, alpha_leaky_relu=0.2, n_heads=4, training=True,
            num_nodes=self.state_size,device=self.device
        )
        self.critic = GATCritic(
            nfeat=1, nhid=16, noutput=1, dropout=0, alpha_leaky_relu=0.2, n_heads=4, training=True,
            num_nodes=self.state_size,device=self.device
        )
        """if self.recurrent:
            self.actor = LSTMActor(state_size=self.state_size, action_size=self.action_size,
                                   hidden_size=self.actor_hidden_size).to(self.device)
            self.critic = LSTMCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size).to(self.device)

        else:
            self.actor = MLPActor(state_size=self.state_size, action_size=self.action_size,
                                  hidden_size=self.actor_hidden_size).to(self.device)
            self.critic = MLPCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size).to(self.device)"""

        print('Initializing discrete PPO agent')
        self.initialize_optimizer()
        # write the hyperparameters

    def choose_action(self, state: np.ndarray,mask_env,deterministic=False) -> (int, torch.Tensor):
        """Choose an action based on the current state

        Arguments
        --------
        state: np.ndarray
            The current state of the environment

        Returns
        -------
        action: int
            The action to take
        log_prob: torch.Tensor
            The log probability of the action
        """
        with torch.no_grad():
            """state = torch.tensor(
                state, device=self.device, dtype=torch.float32)"""
            if self.recurrent:
                state = state.unsqueeze(0)

            action_probs = self.actor(state)
            # Compute the mask
            mask = torch.tensor(mask_env, device=self.device, dtype=torch.float32)
            #mask = self.get_mask(state)

            # Mask the action probabilities
            action_probs = action_probs * mask

            dist = torch.distributions.Categorical(action_probs)
            if deterministic:
                action = torch.argmax(action_probs)
            else:

                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def update(self):
        """Update the policy and value parameters using the PPO algorithm"""
        torch.autograd.set_detect_anomaly(True)

        for _ in tqdm(range(self.epochs)):
            num_samples = len(self.buffer.rewards) - 1
            indices = torch.randperm(num_samples)

            for i in range(0, num_samples, self.minibatch_size):
                batch_indices = indices[i:i + self.minibatch_size]
                states, actions, old_log_probs, advantages, discounted_rewards,masks_list = self.buffer.get_minibatch(
                    batch_indices)

                #states = torch.stack(states)
                """if self.recurrent:
                    states = states.unsqueeze(1)"""
                values_list = []
                action_probs_list= []
                """masks_list = [self.get_mask(
                    self.env.action_space.n) for state in states]"""
                for state in states:
                    values_list.append(self.critic(state).squeeze())
                    action_probs_list.append(self.actor(state))


                #values = values.squeeze().squeeze() if self.recurrent else values.squeeze()

                time_for_preprocessing = time.time()
                # Compute the mask
                action_probs = torch.stack(action_probs_list)
                values = torch.stack(values_list)
                # time before
                masks = np.array(masks_list)
                masks = torch.tensor(masks, device=self.device, dtype=torch.float32)
                #time after
                action_probs = action_probs * masks

                dist = torch.distributions.Categorical(action_probs)
                entropy = dist.entropy()
                discounted_rewards = torch.stack(discounted_rewards)
                discounted_rewards = discounted_rewards.squeeze().squeeze()
                actions = torch.stack(actions)
                actions = actions.squeeze()

                new_log_probs = dist.log_prob(actions)
                advantages = torch.stack(advantages)
                advantages = torch.squeeze(advantages)
                old_log_probs = torch.stack(old_log_probs).squeeze()

                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip,
                                    1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2)

                critic_loss = self.critic_loss(values, discounted_rewards)

                loss = actor_loss + self.value_loss_coef * \
                    critic_loss - self.entropy_coef * entropy
                self.writer.add_scalar(
                    "Value Loss", critic_loss.mean(), self.total_updates_counter)
                self.writer.add_scalar(
                    "MLPActor Loss", actor_loss.mean(), self.total_updates_counter)
                self.writer.add_scalar("Entropy", entropy.mean(
                ) * self.entropy_coef, self.total_updates_counter)
                self.total_updates_counter += 1
                time_after = time.time()
                print(f'preprocessing time: {time_after - time_for_preprocessing}')
                time_before = time.time()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.mean().backward()
                # print the gradients on the actor, for all parameters


                # After the backward call

                self.actor_optimizer.step()
                self.critic_optimizer.step()
                print(f'backward time: {time.time() - time_before}')
                # Update steps here...

        self.decay_learning_rate()
        self.buffer.clean_buffer()

from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F

from src.base.meta_agent.meta_agent import MetaAgent
from src.vanilla_mediator.mediator.actor_critic_mediator import ActorMediator, CriticMediator
import wandb


class Mediator(MetaAgent, ABC):
    def __init__(self, input_sizes, agent_nn, cfg_agent, cfg, agent_i):
        self.n_agents = cfg.env.n_agents
        super(Mediator, self).__init__(input_sizes, [ActorMediator, CriticMediator], cfg_agent=cfg_agent, cfg=cfg)

    def get_policy(self, obs):
        _, pi_dist = self.actor(obs)

        return pi_dist

    def step(self, obs):
        action, _ = self.actor(obs)

        return action.squeeze(0).numpy()

    def update_mediator(self, state, actions, reward, next_state, coalition, done, entropy_coeff):
        mediator_actor_loss_global = []
        update_required = False

        mediator_critic_loss, advantage = self.critic_loss(state, next_state, reward, done)

        for i in range(self.n_agents):
            # filter only states where the mediator was chosen
            idx_mediator = np.where(actions[:, i] != -1)[0]

            if len(idx_mediator) == 0:
                continue

            who = torch.full(idx_mediator.shape, i)
            agent_i = F.one_hot(who, self.n_agents)
            obs = torch.cat([state[idx_mediator], agent_i], dim=-1)

            # choose input only for actor
            advantage_single = (advantage[idx_mediator] * coalition[idx_mediator]).sum(axis=-1, keepdims=True)

            policy = self.get_policy(obs)
            log_prob = policy.log_prob(actions[idx_mediator, i]).unsqueeze(-1)

            # entropy
            entropy = policy.entropy().mean()
            mediator_actor_loss = self.actor_loss(log_prob, advantage_single)

            mediator_actor_loss_global.append((mediator_actor_loss - entropy_coeff * entropy))
            update_required = True

        if not update_required:  # if any meta_agent have chosen Mediator then update
            return

        # make a gradient update
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()

        mediator_actor_loss_global = torch.cat(mediator_actor_loss_global)

        wandb.log({"mediator policy loss": mediator_actor_loss_global.mean().item()})
        wandb.log({"mediator critic loss": mediator_critic_loss.item()})

        mediator_actor_loss_global.mean().backward()

        mediator_critic_loss.backward()

        self.opt_actor.step()
        self.opt_critic.step()

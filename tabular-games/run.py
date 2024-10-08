# import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import argparse

from env.env import IterativePrisonersDilemma, Dilemma
from src.n_step_threat.controller.controller import EyeOfGodNStep
from src.vanilla_mediator.controller.controller import EyeOfGodVanilla

from utils_pkg.wandb_notion import commit_to_notion
import os

# os.environ['WANDB_MODE'] = 'disabled'

# @hydra.main(config_path='conf', config_name='config')
def train(cfg):
    # cfg_notion = cfg.notion
    # cfg = cfg.type
    cfg_wb = OmegaConf.to_container(cfg, resolve=True)
    # wandb.init(project='Mediated_MARL_PD_vanilla', group='test', config=cfg_wb, job_type=cfg.type, mode='online')

    if cfg.name == 'vanilla':
        print(f'Training vanilla mediator in PD')
        env = Dilemma(horizon=cfg.env.horizon)
        zog = EyeOfGodVanilla(cfg)
    elif cfg.name == 'n_step':
        print(f'Training {cfg.env.steps_commit}-step mediator...', end='\n\n')
        env = IterativePrisonersDilemma()
        zog = EyeOfGodNStep(cfg)
        # zog = EyeOfGodVanilla(cfg)
    else:
        print('Specify the type of environment!')
        return -1

    # if cfg_notion.page is not None:
    #     commit_to_notion(cfg_notion.page, wandb.run.get_url(), '/Users/Ilya.Zisman/work/mediator/tabular-games/utils/default.json')
    #     # due to Hydra we need the absolute path here. Unfortunately.

    zog.train(env, True)
    # print(zog)
    # eval_episodes = cfg.env.eval_episodes

    # info = zog.evaluate_policy(eval_episodes, env)
    # mean_step_reward, pick_mediator, policy_agents, policy_mediator, value_agents, value_mediator
    # return info
    return 

if __name__ == '__main__':
    config = OmegaConf.load('conf/type/vanilla.yaml')
    batch_size = config.env.batch_size
    hidden_size = config.agent.n_hidden
    lr_a_agent = config.agent.lr_a
    lr_c_agent = config.agent.lr_c
    lr_a_mediator = config.mediator.lr_a
    lr_c_mediator = config.mediator.lr_c
    entropy_coef = config.agent.entropy_coef
    nIterationd = config.env.iterations
    horizon = config.env.horizon
    num_iter_per_batch = config.env.num_iter_per_batch
    group = f'horizon={horizon}_entropy={entropy_coef}_batch_size={batch_size}_hidden_size={hidden_size}_allr_a={lr_a_agent}_allr_c={lr_c_agent}_nIterations={nIterationd}'
    wandb.init(project='{}update_Mediated_MARL_IPC_vanilla'.format(str(num_iter_per_batch)), group=group, dir='/scratch/ssd004/scratch/shuhui/commitment/')
    train(config)
    wandb.finish()

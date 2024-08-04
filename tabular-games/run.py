import hydra
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


@hydra.main(config_path='conf', config_name='config')
def train(cfg):
    # hydra could override the config with the one from the command line
    # define the parser inside the function to avoid the conflict
    parser = argparse.ArgumentParser()
    parser.add_argument('--array_task_id', type=int, default=0)
    group = 'vanilla'
    name = group + "_" + str(parser.parse_args().array_task_id)
    wandb.init(project='Mediated_MARL_PD_vanilla', group=group, mode='online', name=name)

    cfg_notion = cfg.notion
    cfg = cfg.type
    cfg_wb = OmegaConf.to_container(cfg, resolve=True)
    # wandb.init(project='Mediated_MARL_PD_vanilla', group='test', config=cfg_wb, job_type=cfg.type, mode='online')

    if cfg.name == 'vanilla':
        print(f'Training vanilla mediator in PD')
        env = Dilemma()
        zog = EyeOfGodVanilla(cfg)
    elif cfg.name == 'n_step':
        print(f'Training {cfg.env.steps_commit}-step mediator...', end='\n\n')
        env = IterativePrisonersDilemma()
        zog = EyeOfGodNStep(cfg)
        # zog = EyeOfGodVanilla(cfg)
    else:
        print('Specify the type of environment!')
        return -1

    if cfg_notion.page is not None:
        commit_to_notion(cfg_notion.page, wandb.run.get_url(), '/Users/Ilya.Zisman/work/mediator/tabular-games/utils/default.json')
        # due to Hydra we need the absolute path here. Unfortunately.

    zog.train(env, True)
    # print(zog)
    eval_episodes = cfg.env.eval_episodes

    info = zog.evaluate_policy(eval_episodes, env)
    # mean_step_reward, pick_mediator, policy_agents, policy_mediator, value_agents, value_mediator
    wandb.finish()
    return info

if __name__ == '__main__':
    train()

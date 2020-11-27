    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from utils.shared_buffer import SharedReplayBuffer
from utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.share_policy = self.all_args.share_policy
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_gae = self.all_args.use_gae
        self.gamma = self.all_args.gamma
        self.gae_lambda = self.all_args.gae_lambda
        self.use_proper_time_limits = self.all_args.use_proper_time_limits
        self.use_popart = self.all_args.use_popart
        self.use_single_network = self.all_args.use_single_network

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if "mappo" in self.algorithm_name:
            if self.use_single_network:
                from algorithms.r_mappo_single.r_mappo_single import R_MAPPO as TrainAlgo
                from algorithms.r_mappo_single.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            else:
                from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
                from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif "mappg" in self.algorithm_name:
            if self.use_single_network:
                from algorithms.r_mappg_single.r_mappg_single import R_MAPPG as TrainAlgo
                from algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
            else:
                from algorithms.r_mappg.r_mappg import R_MAPPG as TrainAlgo
                from algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
        else:
            raise NotImplementedError

        if self.share_policy:            
            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
            
            # policy network
            if self.all_args.model_dir == None or self.all_args.model_dir == "":
                policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                device = self.device,
                                cat_self = False if self.use_obs_instead_of_state else True)
            else:
                policy = torch.load(str(self.all_args.model_dir) + "/agent_model.pt")['model']

            # algorithm
            self.trainer = TrainAlgo(self.all_args, policy, device = self.device)
            
            # buffer
            self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])
        else:
            self.trainer = []
            for agent_id in range(self.num_agents):
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
                # policy network
                if self.all_args.model_dir == None or self.all_args.model_dir == "":
                    po = Policy(self.all_args,
                                self.envs.observation_space[agent_id],
                                share_observation_space,
                                self.envs.action_space[agent_id],
                                device = self.device,
                                cat_self = False if self.use_obs_instead_of_state else True)
                else:
                    po = torch.load(str(self.all_args.model_dir) +
                                    "/agent" + str(agent_id) + "_model.pt")['model']
                # algorithm
                tr = TrainAlgo(self.all_args, po, device = self.device)
                self.trainer.append(tr)

            # buffer
            self.buffer = SharedReplayself.buffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[agent_id],
                                        share_observation_space,
                                        self.envs.action_space[agent_id])

    def run(self):

        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads)
        last_battles_won = np.zeros(self.n_rollout_threads)

        for episode in range(episodes):
            if self.use_linear_lr_decay:  # decrease learning rate linearly
                if self.share_policy:
                    self.trainer.policy.lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, recurrent_hidden_statess, recurrent_hidden_statess_critic = self.collect_data(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       recurrent_hidden_statess, recurrent_hidden_statess_critic 
                
                # insert data into buffer
                self.insert_data(data)

            # compute return and update network
            stats = self.update_network()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                
                self.log(stats, total_num_steps)

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(
                                info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(
                                info[0]['battles_game']-last_battles_game[i])

                    if np.sum(incre_battles_game) > 0:
                        if self.use_wandb:
                            wandb.log({"incre_win_rate": np.sum(
                                incre_battles_won)/np.sum(incre_battles_game)}, step=total_num_steps)
                        else:
                            self.writter.add_scalars("incre_win_rate", {"incre_win_rate": np.sum(
                                incre_battles_won)/np.sum(incre_battles_game)}, total_num_steps)
                        print("incre win rate is {}.".format(
                            np.sum(incre_battles_won)/np.sum(incre_battles_game)))
                    else:
                        if self.use_wandb:
                            wandb.log({"incre_win_rate": 0}, step=total_num_steps)
                        else:
                            self.writter.add_scalars("incre_win_rate", {
                                                "incre_win_rate": 0}, total_num_steps)
                    last_battles_game = battles_game
                    last_battles_won = battles_won

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1) if self.use_centralized_V else obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect_data(self, step):
        if self.share_policy:
            self.trainer.prep_rollout()
            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                = self.trainer.policy.act(torch.from_numpy(np.concatenate(self.buffer.share_obs[step])),
                                torch.from_numpy(np.concatenate(self.buffer.obs[step])),
                                torch.from_numpy(np.concatenate(self.buffer.recurrent_hidden_states[step])),
                                torch.from_numpy(np.concatenate(self.buffer.recurrent_hidden_states_critic[step])),
                                torch.from_numpy(np.concatenate(self.buffer.masks[step])),
                                torch.from_numpy(np.concatenate(self.buffer.available_actions[step])))
            # [self.envs, agents, dim]
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            recurrent_hidden_statess = np.array(np.split(_t2n(recurrent_hidden_states), self.n_rollout_threads))
            recurrent_hidden_statess_critic = np.array(np.split(_t2n(recurrent_hidden_states_critic), self.n_rollout_threads))
        else:
            values = []
            actions = []
            action_log_probs = []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []

            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                    = self.trainer[agent_id].policy.act(torch.from_numpy(self.buffer.share_obs[step, :, agent_id]),
                                                torch.from_numpy(self.buffer.obs[step, :, agent_id]),
                                                torch.from_numpy(self.buffer.recurrent_hidden_states[step, :, agent_id]),
                                                torch.from_numpy(self.buffer.recurrent_hidden_states_critic[step, :, agent_id]),
                                                torch.from_numpy(self.buffer.masks[step, :, agent_id]),
                                                torch.from_numpy(self.buffer.available_actions[step, :, agent_id]))

                values.append(_t2n(value))
                actions.append(_t2n(action))
                action_log_probs.append(_t2n(action_log_prob))
                recurrent_hidden_statess.append(_t2n(recurrent_hidden_states))
                recurrent_hidden_statess_critic.append(_t2n(recurrent_hidden_states_critic))

            values = np.array(values).transpose(1, 0, 2)
            actions = np.array(actions).transpose(1, 0, 2)
            action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
            recurrent_hidden_statess = np.array(recurrent_hidden_statess).transpose(1, 0, 2)
            recurrent_hidden_statess_critic = np.array(recurrent_hidden_statess_critic).transpose(1, 0, 2)

        return values, actions, action_log_probs, recurrent_hidden_statess, recurrent_hidden_statess_critic

    def insert_data(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, recurrent_hidden_statess, recurrent_hidden_statess_critic = data

        dones_env = np.all(dones, axis=1)

        recurrent_hidden_statess[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        recurrent_hidden_statess_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = []
        for info in infos:
            bad_mask = []
            active_mask = []
            for agent_id in range(self.num_agents):
                if info[agent_id]['bad_transition']:
                    bad_mask.append([0.0])
                else:
                    bad_mask.append([1.0])
            bad_masks.append(bad_mask)

        if self.use_centralized_V:
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, recurrent_hidden_statess, recurrent_hidden_statess_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def update_network(self):
        if self.share_policy:
            with torch.no_grad():
                self.trainer.prep_rollout()
                next_value = self.trainer.policy.get_value(torch.from_numpy(np.concatenate(self.buffer.share_obs[-1])),
                                                        torch.from_numpy(np.concatenate(self.buffer.recurrent_hidden_states_critic[-1])),
                                                        torch.from_numpy(np.concatenate(self.buffer.masks[-1])))
                next_values = np.array(np.split(_t2n(next_value), self.n_rollout_threads))
                self.buffer.shared_compute_returns(next_values,
                                                    self.use_gae,
                                                    self.gamma,
                                                    self.gae_lambda,
                                                    self.use_proper_time_limits,
                                                    self.use_popart,
                                                    self.trainer.value_normalizer)
            # update network
            self.trainer.prep_training()
            stats = self.trainer.shared_update(self.buffer)

        else:
            value_losses = []
            action_losses = []
            dist_entropies = []
            actor_grad_norms = []
            critic_grad_norms = []
            joint_losses = []
            joint_grad_norms = []

            for agent_id in range(self.num_agents):
                with torch.no_grad():
                    self.trainer[agent_id].prep_rollout()
                    next_value = self.trainer[agent_id].policy.get_value(torch.from_numpy(self.buffer.share_obs[-1, :, agent_id]), 
                                                                torch.from_numpy(self.buffer.recurrent_hidden_states_critic[-1, :, agent_id]),
                                                                torch.from_numpy(self.buffer.masks[-1, :, agent_id]))
                    next_value = _t2n(next_value)
                    self.buffer.single_compute_returns(agent_id,
                                                        next_value,
                                                        self.use_gae,
                                                        self.gamma,
                                                        self.gae_lambda,
                                                        self.use_proper_time_limits,
                                                        self.use_popart,
                                                        self.trainer[agent_id].value_normalizer)
                # update network
                actor_critic[agent_id].prep_training()
                value_loss, critic_grad_norm, action_loss, dist_entropy, actor_grad_norm, joint_loss, joint_grad_norm = self.trainer[agent_id].single_update(
                    agent_id, self.buffer)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                actor_grad_norms.append(actor_grad_norm)
                critic_grad_norms.append(critic_grad_norm)
                joint_losses.append(joint_loss)
                joint_grad_norms.append(joint_grad_norm)

            stats = value_losses, critic_grad_norms, action_losses, dist_entropies, actor_grad_norms, joint_losses, joint_grad_norms
        
        self.buffer.after_update()
        return stats

    def save(self):
        if self.share_policy:
            torch.save({
                'model': self.trainer.policy
            },
                self.save_dir + "/agent_model.pt")
        else:
            for agent_id in range(self.num_agents):
                torch.save({
                    'model': self.trainer[agent_id].policy
                },
                    self.save_dir + "/agent%i_model" % agent_id + ".pt")

    def log(self, stats, total_num_steps): 
        if self.share_policy:
            value_loss, critic_grad_norm, action_loss, dist_entropy, actor_grad_norm, joint_loss, joint_grad_norm = stats
            print("value loss of agent: " + str(value_loss))
            if self.use_wandb:
                wandb.log({"value_loss": value_loss}, step=total_num_steps)
                wandb.log({"action_loss": action_loss},
                        step=total_num_steps)
                wandb.log({"dist_entropy": dist_entropy},
                        step=total_num_steps)
                wandb.log({"actor_grad_norm": actor_grad_norm}, step=total_num_steps)
                wandb.log({"critic_grad_norm": critic_grad_norm}, step=total_num_steps)
                if "mappg" in self.algorithm_name:
                    wandb.log({"joint_loss": joint_loss}, step=total_num_steps)
                    wandb.log({"joint_grad_norm": joint_grad_norm}, step=total_num_steps)
                wandb.log({"average_step_rewards": np.mean(
                    self.buffer.rewards)}, step=total_num_steps)
            else:
                self.writter.add_scalars(
                    "value_loss", {"value_loss": value_loss}, total_num_steps)
                self.writter.add_scalars(
                    "action_loss", {"action_loss": action_loss}, total_num_steps)
                self.writter.add_scalars(
                    "dist_entropy", {"dist_entropy": dist_entropy}, total_num_steps)
                self.writter.add_scalars(
                    "actor_grad_norm", {"actor_grad_norm": actor_grad_norm}, total_num_steps)
                self.writter.add_scalars(
                    "critic_grad_norm", {"critic_grad_norm": critic_grad_norm}, total_num_steps)
                if "mappg" in self.algorithm_name:
                    self.writter.add_scalars(
                        "joint_loss", {"joint_loss": joint_loss}, total_num_steps)
                    self.writter.add_scalars(
                        "joint_grad_norm", {"joint_grad_norm": joint_grad_norm}, total_num_steps)
                self.writter.add_scalars("average_step_rewards", {
                                    "average_step_rewards": np.mean(self.buffer.rewards)}, total_num_steps)
        else:
            value_losses, critic_grad_norms, action_losses, dist_entropies, actor_grad_norms, joint_losses, joint_grad_norms = stats
            for agent_id in range(self.num_agents):
                print("value loss of agent%i: " % agent_id + str(value_losses[agent_id]))
                if self.use_wandb:
                    wandb.log(
                        {"agent%i/value_loss" % agent_id: value_losses[agent_id]}, step=total_num_steps)
                    wandb.log(
                        {"agent%i/action_loss" % agent_id: action_losses[agent_id]}, step=total_num_steps)
                    wandb.log(
                        {"agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, step=total_num_steps)
                    wandb.log(
                        {"agent%i/actor_grad_norm" % agent_id: actor_grad_norms[agent_id]}, step=total_num_steps)
                    wandb.log(
                        {"agent%i/critic_grad_norm" % agent_id: critic_grad_norms[agent_id]}, step=total_num_steps)
                    if "mappg" in self.algorithm_name:
                        wandb.log(
                            {"agent%i/joint_loss" % agent_id: joint_losses[agent_id]}, step=total_num_steps)
                        wandb.log(
                            {"agent%i/joint_grad_norm" % agent_id: joint_grad_norms[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/average_step_rewards" % agent_id: np.mean(self.buffer.rewards[:, :, agent_id])}, step=total_num_steps)
                else:
                    self.writter.add_scalars("agent%i/value_loss" % agent_id, {
                                        "agent%i/value_loss" % agent_id: value_losses[agent_id]}, total_num_steps)
                    self.writter.add_scalars("agent%i/action_loss" % agent_id, {
                                        "agent%i/action_loss" % agent_id: action_losses[agent_id]}, total_num_steps)
                    self.writter.add_scalars("agent%i/dist_entropy" % agent_id, {
                                        "agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, total_num_steps)
                    self.writter.add_scalars("agent%i/actor_grad_norm" % agent_id, {
                                        "agent%i/actor_grad_norm" % agent_id: actor_grad_norms[agent_id]}, total_num_steps)
                    self.writter.add_scalars("agent%i/critic_grad_norm" % agent_id, {
                                        "agent%i/critic_grad_norm" % agent_id: critic_grad_norms[agent_id]}, total_num_steps)
                    if "mappg" in self.algorithm_name:
                        self.writter.add_scalars("agent%i/joint_loss" % agent_id, {
                                            "agent%i/joint_loss" % agent_id: joint_losses[agent_id]}, total_num_steps)
                        self.writter.add_scalars("agent%i/joint_grad_norm" % agent_id, {
                                            "agent%i/joint_grad_norm" % agent_id: joint_grad_norms[agent_id]}, total_num_steps)
                    self.writter.add_scalars("agent%i/average_step_rewards" % agent_id, {
                                        "agent%i/average_step_rewards" % agent_id: np.mean(self.buffer.rewards[:, :, agent_id])}, total_num_steps)

    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        if self.use_centralized_V:
            eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            eval_share_obs = eval_obs

        eval_recurrent_hidden_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.hidden_size), dtype=np.float32)
        eval_recurrent_hidden_states_critic = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            if self.share_policy:
                self.trainer.prep_rollout()
                _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = self.trainer.policy.act(torch.from_numpy(np.concatenate(eval_share_obs)),
                                                                                                                    torch.from_numpy(
                                                                                                                        np.concatenate(eval_obs)),
                                                                                                                    torch.from_numpy(np.concatenate(
                                                                                                                        eval_recurrent_hidden_states)),
                                                                                                                    torch.from_numpy(np.concatenate(
                                                                                                                        eval_recurrent_hidden_states_critic)),
                                                                                                                    torch.from_numpy(
                                                                                                                        np.concatenate(eval_masks)),
                                                                                                                    torch.from_numpy(np.concatenate(
                                                                                                                        eval_available_actions)),
                                                                                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                eval_recurrent_hidden_states = np.array(np.split( _t2n(eval_recurrent_hidden_state), self.n_eval_rollout_threads))
                eval_recurrent_hidden_states_critic = np.array(np.split(_t2n(eval_recurrent_hidden_state_critic), self.n_eval_rollout_threads))
            else:
                eval_actions = []
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = self.trainer[agent_id].policy.act(torch.from_numpy(eval_share_obs[:, agent_id]),
                                                                                                                                    torch.from_numpy(
                                                                                                                                        eval_obs[:, agent_id]),
                                                                                                                                    torch.from_numpy(
                                                                                                                                        eval_recurrent_hidden_states[:, agent_id]),
                                                                                                                                    torch.from_numpy(
                                                                                                                                        eval_recurrent_hidden_states_critic[:, agent_id]),
                                                                                                                                    torch.from_numpy(eval_masks[:, agent_id]),
                                                                                                                                    torch.from_numpy(
                                                                                                                                        eval_available_actions[:, agent_id, :]),
                                                                                                                                    deterministic=True)

                    eval_actions.append(_t2n(eval_action))
                    eval_recurrent_hidden_states[:, agent_id] = _t2n(eval_recurrent_hidden_state)
                    eval_recurrent_hidden_states_critic[:, agent_id] = _t2n(eval_recurrent_hidden_state_critic)

                eval_actions = np.array(eval_actions).transpose(1, 0, 2)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            if self.use_centralized_V:
                eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(self.num_agents, axis=1)
            else:
                eval_share_obs = eval_obs

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_recurrent_hidden_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
            eval_recurrent_hidden_states_critic[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones( (self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                print("eval win rate is {}.".format(eval_battles_won/eval_episode))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_battles_won/eval_episode}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_battles_won/eval_episode}, total_num_steps)
                break
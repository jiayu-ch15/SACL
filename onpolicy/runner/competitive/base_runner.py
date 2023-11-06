from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
import os
import psutil
# import slackweb
import socket
import torch
import wandb

from onpolicy.utils.shared_buffer import SharedReplayBuffer


webhook_url = "https://hooks.slack.com/services/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"

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]       
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        self.num_red = config["num_red"]
        self.num_blue = config["num_blue"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.eval_episodes = self.all_args.eval_episodes
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_single_network = self.all_args.use_single_network
        self.use_valuenorm = self.all_args.use_valuenorm
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # training mode: self-play, red br, blue br
        self.training_mode = self.all_args.training_mode
        self.train_red = (self.training_mode in ["self_play", "red_br"])
        self.train_blue = (self.training_mode in ["self_play", "blue_br"])
        self.red_model_dir = self.all_args.red_model_dir
        self.blue_model_dir = self.all_args.blue_model_dir
        self.red_valuenorm_dir = self.all_args.red_valuenorm_dir
        self.blue_valuenorm_dir = self.all_args.blue_valuenorm_dir
        # if self.training_mode == "red_br":
        #     assert (self.blue_model_dir is not None), ("to train a red br, you should set the value of --blue_model_dir")
        # if self.training_mode == "blue_br":
        #     assert (self.red_model_dir is not None), ("to train a blue br, your should set the value of --red_model_dir")

        # interval
        self.save_interval = self.all_args.save_interval
        self.save_ckpt_interval = self.all_args.save_ckpt_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.opponent_name = self.all_args.oppenent_name

        if self.use_render:
            self.gif_dir = f"{self.red_model_dir[0]}/gifs"
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / "logs")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / "models")
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
        
        if "mappo" in self.algorithm_name:
            if self.use_single_network:
                from onpolicy.algorithms.r_mappo_single.r_mappo_single import R_MAPPO as TrainAlgo
                from onpolicy.algorithms.r_mappo_single.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            else:
                from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
                from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif "mappg" in self.algorithm_name:
            if self.use_single_network:
                from onpolicy.algorithms.r_mappg_single.r_mappg_single import R_MAPPG as TrainAlgo
                from onpolicy.algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
            else:
                from onpolicy.algorithms.r_mappg.r_mappg import R_MAPPG as TrainAlgo
                from onpolicy.algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
        elif self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        # elif self.algorithm_name == "happo":
        #     from onpolicy.algorithms.happo.happo_trainer import HAPPO as TrainAlgo
        #     from onpolicy.algorithms.happo.policy import HAPPO_Policy as Policy
        elif "ft" in self.algorithm_name:
            print("use frontier-based algorithm")
        else:
            raise NotImplementedError

        if self.algorithm_name == self.opponent_name:
            # self-play
            if self.algorithm_name == "mat":
                # policy network
                self.red_policy = Policy(
                    self.all_args,
                    self.envs.observation_space[0],
                    self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0],
                    self.envs.action_space[0],
                    num_agents=self.num_red,
                    device=self.device,
                )
                self.blue_policy = Policy(
                    self.all_args,
                    self.envs.observation_space[-1],
                    self.envs.share_observation_space[-1] if self.use_centralized_V else self.envs.observation_space[-1],
                    self.envs.action_space[-1],
                    num_agents=self.num_blue,
                    device=self.device,
                )
            else:
                # policy network
                self.red_policy = Policy(
                    self.all_args,
                    self.envs.observation_space[0],
                    self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0],
                    self.envs.action_space[0],
                    device=self.device,
                )
                self.blue_policy = Policy(
                    self.all_args,
                    self.envs.observation_space[-1],
                    self.envs.share_observation_space[-1] if self.use_centralized_V else self.envs.observation_space[-1],
                    self.envs.action_space[-1],
                    device=self.device,
                )

            # algorithm
            if self.algorithm_name == "mat":
                self.red_trainer = TrainAlgo(self.all_args, self.red_policy, num_agents=self.num_red, device=self.device)
                self.blue_trainer = TrainAlgo(self.all_args, self.blue_policy, num_agents=self.num_blue, device=self.device)
            else:
                self.red_trainer = TrainAlgo(self.all_args, self.red_policy, device=self.device)
                self.blue_trainer = TrainAlgo(self.all_args, self.blue_policy, device=self.device)

            # restore model
            if self.red_model_dir is not None:
                self.restore("red_model")
            if self.blue_model_dir is not None:
                self.restore("blue_model")
            if self.use_valuenorm and self.red_valuenorm_dir is not None:
                self.restore("red_valuenorm")
            if self.use_valuenorm and self.blue_valuenorm_dir is not None:
                self.restore("blue_valuenorm")

            # buffer
            self.red_buffer = SharedReplayBuffer(
                self.all_args,
                self.num_red,
                self.envs.observation_space[0],
                self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0],
                self.envs.action_space[0],
            )
            self.blue_buffer = SharedReplayBuffer(
                self.all_args,
                self.num_blue,
                self.envs.observation_space[-1],
                self.envs.share_observation_space[-1] if self.use_centralized_V else self.envs.observation_space[-1],
                self.envs.action_space[-1],
            )
        else:
            # br, set opponent_algorithm_name
            if self.opponent_name == 'mappg':
                from onpolicy.algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as OppoPolicy
            elif self.opponent_name == 'mat':
                from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as OppoPolicy
                # from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy_ensemble as OppoPolicy
            elif self.opponent_name == 'matrpo':
                from onpolicy.algorithms.r_trpo.algorithm.rTRPOPolicy import R_TRPOPolicy as OppoPolicy

            # policy network
            self.red_policy = Policy(
                self.all_args,
                self.envs.observation_space[0],
                self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            if self.opponent_name == 'mat':
                self.blue_policy = OppoPolicy(
                    self.all_args,
                    self.envs.observation_space[-1],
                    self.envs.share_observation_space[-1] if self.use_centralized_V else self.envs.observation_space[-1],
                    self.envs.action_space[-1],
                    num_agents=self.num_blue,
                    device=self.device,
                )
            else:
                self.blue_policy = OppoPolicy(
                    self.all_args,
                    self.envs.observation_space[-1],
                    self.envs.share_observation_space[-1] if self.use_centralized_V else self.envs.observation_space[-1],
                    self.envs.action_space[-1],
                    device=self.device,
                )

            # algorithm
            self.red_trainer = TrainAlgo(self.all_args, self.red_policy, device=self.device)
            self.blue_trainer = TrainAlgo(self.all_args, self.blue_policy, device=self.device, role='blue')

            # restore model
            if self.red_model_dir is not None:
                self.restore("red_model")
            if self.blue_model_dir is not None:
                self.restore("blue_model")
            if self.use_valuenorm and self.red_valuenorm_dir is not None:
                self.restore("red_valuenorm")
            if self.use_valuenorm and self.blue_valuenorm_dir is not None:
                self.restore("blue_valuenorm")

            # buffer
            self.red_buffer = SharedReplayBuffer(
                self.all_args,
                self.num_red,
                self.envs.observation_space[0],
                self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0],
                self.envs.action_space[0],
            )
            self.blue_buffer = SharedReplayBuffer(
                self.all_args,
                self.num_blue,
                self.envs.observation_space[-1],
                self.envs.share_observation_space[-1] if self.use_centralized_V else self.envs.observation_space[-1],
                self.envs.action_space[-1],
            )

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def eval_head2head(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def eval_cross_play(self):
        num_red_models = len(self.red_model_dir)
        num_blue_models = len(self.blue_model_dir)

        cross_play_returns = np.zeros((num_red_models, num_blue_models), dtype=float)
        for red_idx in range(num_red_models):
            self.restore("red_model", red_idx)
            for blue_idx in range(num_blue_models):
                print(f"red model {red_idx} v.s. blue model {blue_idx}")
                self.restore("blue_model", blue_idx)
                cross_play_returns[red_idx, blue_idx] = self.eval_head2head()
        np.save(f"{self.log_dir}/cross_play_returns.npy", cross_play_returns)
    
    @torch.no_grad()
    def compute(self):
        if self.train_red:
            self.red_trainer.prep_rollout()
            red_next_values = self.red_trainer.policy.get_values(
                np.concatenate(self.red_buffer.share_obs[-1]),
                np.concatenate(self.red_buffer.rnn_states_critic[-1]),
                np.concatenate(self.red_buffer.masks[-1]),
            )
            red_next_values = np.array(np.split(_t2n(red_next_values), self.n_rollout_threads))
            self.red_buffer.compute_returns(red_next_values, self.red_trainer.value_normalizer)

        if self.train_blue:
            self.blue_trainer.prep_rollout()
            blue_next_values = self.blue_trainer.policy.get_values(
                np.concatenate(self.blue_buffer.share_obs[-1]),
                np.concatenate(self.blue_buffer.rnn_states_critic[-1]),
                np.concatenate(self.blue_buffer.masks[-1]),
            )
            blue_next_values = np.array(np.split(_t2n(blue_next_values), self.n_rollout_threads))
            self.blue_buffer.compute_returns(blue_next_values, self.blue_trainer.value_normalizer)

    def train(self):
        red_train_infos = None
        if self.train_red:
            self.red_trainer.prep_training()
            red_train_infos = self.red_trainer.train(self.red_buffer)      
        self.red_buffer.after_update()

        blue_train_infos = None
        if self.train_blue:
            self.blue_trainer.prep_training()
            blue_train_infos = self.blue_trainer.train(self.blue_buffer)      
        self.blue_buffer.after_update()

        # self.log_system()
        return red_train_infos, blue_train_infos

    def save(self):
        if self.use_single_network:
            red_policy_model = self.red_trainer.policy.model
            torch.save(red_policy_model.state_dict(), f"{self.save_dir}/red_model.pt")
            blue_policy_model = self.blue_trainer.policy.model
            torch.save(blue_policy_model.state_dict(), f"{self.save_dir}/blue_model.pt")
        else:
            red_policy_actor = self.red_trainer.policy.actor
            torch.save(red_policy_actor.state_dict(), f"{self.save_dir}/red_actor.pt")
            red_policy_critic = self.red_trainer.policy.critic
            torch.save(red_policy_critic.state_dict(), f"{self.save_dir}/red_critic.pt")
            if self.use_valuenorm:
                red_value_normalizer = self.red_trainer.value_normalizer
                torch.save(red_value_normalizer.state_dict(), f"{self.save_dir}/red_value_normalizer.pt")

            if self.opponent_name == 'mat':
                blue_policy = self.blue_trainer.policy.transformer
                torch.save(blue_policy.state_dict(), f"{self.save_dir}/blue_actor.pt")
            else:
                blue_policy_actor = self.blue_trainer.policy.actor
                torch.save(blue_policy_actor.state_dict(), f"{self.save_dir}/blue_actor.pt")
                blue_policy_critic = self.blue_trainer.policy.critic
                torch.save(blue_policy_critic.state_dict(), f"{self.save_dir}/blue_critic.pt")
                if self.use_valuenorm:
                    blue_value_normalizer = self.blue_trainer.value_normalizer
                    torch.save(blue_value_normalizer.state_dict(), f"{self.save_dir}/blue_value_normalizer.pt")
    
    def save_ckpt(self, total_num_steps):
        million_steps = int(total_num_steps // 1000000)
        save_ckpt_dir = f"{self.save_dir}/{million_steps}M"
        assert not os.path.exists(save_ckpt_dir), (f"checkpoint path {save_ckpt_dir} already exists.")
        os.makedirs(save_ckpt_dir)
        
        if self.use_single_network:
            red_policy_model = self.red_trainer.policy.model
            torch.save(red_policy_model.state_dict(), f"{save_ckpt_dir}/red_model.pt")
            blue_policy_model = self.blue_trainer.policy.model
            torch.save(blue_policy_model.state_dict(), f"{save_ckpt_dir}/blue_model.pt")
        else:
            red_policy_actor = self.red_trainer.policy.actor
            torch.save(red_policy_actor.state_dict(), f"{save_ckpt_dir}/red_actor.pt")
            red_policy_critic = self.red_trainer.policy.critic
            torch.save(red_policy_critic.state_dict(), f"{save_ckpt_dir}/red_critic.pt")
            if self.use_valuenorm:
                red_value_normalizer = self.red_trainer.value_normalizer
                torch.save(red_value_normalizer.state_dict(), f"{save_ckpt_dir}/red_value_normalizer.pt")

            if self.opponent_name == 'mat':
                blue_policy = self.blue_trainer.policy.transformer
                torch.save(blue_policy.state_dict(), f"{save_ckpt_dir}/blue_actor.pt")
            else:
                blue_policy_actor = self.blue_trainer.policy.actor
                torch.save(blue_policy_actor.state_dict(), f"{save_ckpt_dir}/blue_actor.pt")
                blue_policy_critic = self.blue_trainer.policy.critic
                torch.save(blue_policy_critic.state_dict(), f"{save_ckpt_dir}/blue_critic.pt")
                if self.use_valuenorm:
                    blue_value_normalizer = self.blue_trainer.value_normalizer
                    torch.save(blue_value_normalizer.state_dict(), f"{save_ckpt_dir}/blue_value_normalizer.pt")

    def restore(self, model, idx=0):
        if model == "red_model":
            if self.use_single_network:
                red_policy_model_state_dict = torch.load(f"{self.red_model_dir[idx]}/red_model.pt", map_location=self.device)
                self.red_trainer.policy.model.load_state_dict(red_policy_model_state_dict)
            else:
                red_policy_actor_state_dict = torch.load(f"{self.red_model_dir[idx]}/red_actor.pt", map_location=self.device)
                # red_policy_critic_state_dict = torch.load(f"{self.red_model_dir[idx]}/red_critic.pt", map_location=self.device)
                self.red_trainer.policy.actor.load_state_dict(red_policy_actor_state_dict)
                # if not (self.all_args.use_render or self.all_args.use_eval):
                # self.red_trainer.policy.critic.load_state_dict(red_policy_critic_state_dict)
        elif model == "red_valuenorm":
            red_value_normalizer_state_dict = torch.load(f"{self.red_valuenorm_dir[idx]}/red_value_normalizer.pt", map_location=self.device)
            self.red_trainer.value_normalizer.load_state_dict(red_value_normalizer_state_dict)
        elif model == "blue_model":
            if self.use_single_network:
                blue_policy_model_state_dict = torch.load(f"{self.blue_model_dir[idx]}/blue_model.pt", map_location=self.device)
                self.blue_trainer.policy.model.load_state_dict(blue_policy_model_state_dict)
            else:
                if self.opponent_name == 'mat':
                    blue_policy_actor_state_dict = torch.load(f"{self.blue_model_dir[idx]}/blue_actor.pt", map_location=self.device)
                    self.blue_trainer.policy.transformer.load_state_dict(blue_policy_actor_state_dict)
                else:
                    blue_policy_actor_state_dict = torch.load(f"{self.blue_model_dir[idx]}/blue_actor.pt", map_location=self.device)
                    import pdb; pdb.set_trace()
                    self.blue_trainer.policy.actor.load_state_dict(blue_policy_actor_state_dict)
                # if not (self.all_args.use_render or self.all_args.use_eval):
                # blue_policy_critic_state_dict = torch.load(f"{self.blue_model_dir[idx]}/blue_critic.pt", map_location=self.device)
                # self.blue_trainer.policy.critic.load_state_dict(blue_policy_critic_state_dict)
        elif model == "blue_valuenorm":
            blue_value_normalizer_state_dict = torch.load(f"{self.blue_valuenorm_dir[idx]}/blue_value_normalizer.pt", map_location=self.device)
            self.blue_trainer.value_normalizer.load_state_dict(blue_value_normalizer_state_dict)
        else:
            raise NotImplementedError(f"Model {model} is not supported.")
 
    def log_train(self, red_train_infos, blue_train_infos, total_num_steps):
        if self.train_red:
            for k, v in red_train_infos.items():
                key = f"red/{k}"
                if self.use_wandb:
                    wandb.log({key: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(key, {key: v}, total_num_steps)

        if self.train_blue:
            for k, v in blue_train_infos.items():
                key = f"blue/{k}"
                if self.use_wandb:
                    wandb.log({key: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(key, {key: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def log_system(self):
        # RRAM
        mem = psutil.virtual_memory()
        total_mem = float(mem.total) / 1024 / 1024 / 1024
        used_mem = float(mem.used) / 1024 / 1024 / 1024
        if used_mem/total_mem > 0.95:
            slack = slackweb.Slack(url=webhook_url)
            host_name = socket.gethostname()
            slack.notify(text="Host {}: occupied memory is *{:.2f}*%!".format(host_name, used_mem/total_mem*100))

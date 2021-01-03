import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed 
        --cuda
            by default, use GPU CUDA; if set, use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default, use wandb. set for local training test (do not upload data to wandb server)
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default, use global state; if set, use concatenated obs.
    Replay Buffer parameters:
        --episode_length <int>
            the number of steps in an single episode
    network parameters:
        --share_policy
            by default, training agents share the same polic; set to make training agents use different policies. 
        --use_centralized_V
            by default, used centralized Value Function; if set, do not use centralized_V, otherwise 
        --use_conv1d
            by default, do not use conv1d. If set, use conv1d
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default, use relu. If set, eliminate ReLu
        --use_popart
            by default, use  # TODO @zoeyuchao. The same comment might in need of change.
        --use_feature_popart
            by default, do not use popart to inputs. if set, apply popart to inputs. # TODO @zoeyuchao. The same comment might in need of change.
        --use_feature_normalization
            by default, apply layernorm to the inputs 
        --use_orthogonal
            by default, use Orthogonal initialization for weights and 0 initialization for biases. If set, do not use Orthogonal inilialization.
        --gain
            by default, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default, do not use naive version, if set, use a naive recurrent policy
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent Network? (only support 1 for now, default 1). 
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
        --use_attn
            by default, use attention tactics. # TODO @zoeyuchao. 
        --attn_N
            the number of ??? by default 1,  # TODO @zoeyuchao. 
        --attn_size
            by default, use # TODO @zoeyuchao. 
        --attn_heads
            by default, use # TODO @zoeyuchao. 
        --dropout
            by default, use  # TODO @zoeyuchao. 
        --use_average_pool 
            by default, do not use average pooling. If set, use average pooling
        --use_cat_self
            by default, use  # TODO @zoeyuchao. 
    Optimizer Parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0) # TODO @zoeyuchao. Not sure about the meaning
    
    PPO Parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_policy_vhead
            by default, do not use policy vhead. if set, use policy vhead.
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --policy_value_loss_coef <float>
            policy value loss coefficient (default: 0.5)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default, true  # TODO @zoeyuchao. 
        --use_return_active_masks
            by default, true  # TODO @zoeyuchao. 
        --huber_delta <float>
            set value of huber delta   # TODO @zoeyuchao. 
        --aux_epoch <int>
            number of auxiliary epochs (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
        --use_single_network
            by default, do not use centralized V function.    # TODO @zoeyuchao. Difference with use_centralized_V ?
    run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    Eval Parameters:
        --use_eval
            by default, do not start evaluation. If set, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    Pretrained Parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='mappo', choices=["rmappo", "mappo", "rmappg", "mappg"])

    parser.add_argument("--experiment_name", type=str, default="check")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True)
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True)
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollout")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollout")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollout")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='yuchao')
    parser.add_argument("--use_wandb", action='store_false', default=True)

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2')
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") # TODO @zoeyuchao. The same comment might in need of change.
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_false', default=True)
    parser.add_argument("--use_feature_popart", action='store_true',
                        default=False, help="Whether to apply popart to the inputs")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    # TODO now only 1 is support
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # attn parameters
    parser.add_argument("--use_attn", action='store_true', default=False)
    parser.add_argument("--attn_N", type=int, default=1)
    parser.add_argument("--attn_size", type=int, default=64)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_average_pool",
                        action='store_false', default=True)
    parser.add_argument("--use_cat_self", action='store_false', default=True)

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_policy_vhead",
                        action='store_true', default=False)
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True)
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--policy_value_loss_coef", type=float,
                        default=1, help='policy value loss coefficient (default: 0.5)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True)
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True)
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True)
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True)
    parser.add_argument("--huber_delta", type=float, default=10.0)

    # ppg parameters
    parser.add_argument("--aux_epoch", type=int, default=5,
                        help='number of auxiliary epochs (default: 4)')
    parser.add_argument("--clone_coef", type=float, default=1.0,
                        help='clone term coefficient (default: 0.01)')
    parser.add_argument("--use_single_network", action='store_true',
                        default=False, help="Whether to use centralized V function")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')

    # save parameters
    parser.add_argument("--save_interval", type=int, default=50)

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5)

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--eval_episodes", type=int, default=32)

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False)
    parser.add_argument("--use_render", action='store_true', default=False)
    parser.add_argument("--render_episodes", type=int, default=5)
    parser.add_argument("--ifi", type=float, default=0.1)

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None)

    return parser

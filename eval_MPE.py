
import argparse
import pickle
import copy
import glob
from pathlib import Path
import os, time
import numpy as np

import torch

from utils.make_env import make_env
import imageio

from config import get_config
from utils.storage import RolloutStorage   

def cover_landmark(agent):
    dist = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos)))
    if dist <= agent.size + agent.goal.size:
        return True
    else:
        return False    

if __name__ == '__main__':

    args = get_config()
    assert args.env_name == "single_navigation", ("only MPE is supported in this file, check the config.py.")
    assert args.num_agents == 1, ("only 1 agent is supported in single_navigation env, check the config.py.")

    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    
    gif_dir = Path('./gifs') / args.algorithm_name
    if not os.path.exists(str(gif_dir)):
        os.makedirs(str(gif_dir))

    env = make_env(args)

    actor_critic = []
    for i in range(args.num_agents):
        ac = torch.load(args.model_dir + "/agent%i_model" % i + ".pt")['model'].to(device)
        actor_critic.append(ac)
        
    cover = 0

    for episode in range(args.eval_episodes):
        print("Episode %i of %i" % (episode, args.eval_episodes))
        if args.save_gifs:
            frames = []
            #env.render('human')
            frames.append(env.render('rgb_array')[0])
        
        state = env.reset()
        state = np.array([state])

        share_obs = []
        obs = []
        recurrent_hidden_statess = []
        recurrent_hidden_statess_critic = []
        recurrent_c_statess = []
        recurrent_c_statess_critic = []
        masks = []
        
        # rollout
        for i in range(args.num_agents):
            if len(env.observation_space[0].shape) == 1:
                share_obs.append((torch.tensor(state.reshape(1, -1),dtype=torch.float32)).to(device))
                obs.append((torch.tensor(state[:,i,:],dtype=torch.float32)).to(device))
                recurrent_hidden_statess.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))
                recurrent_hidden_statess_critic.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size * args.num_agents).to(device))
                recurrent_c_statess.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))
                recurrent_c_statess_critic.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size * args.num_agents).to(device))
                masks.append(torch.ones(1,1).to(device))
            else:
                raise NotImplementedError

        for step in range(args.episode_length):
            print("step %i of %i" % (step, args.episode_length))
            calc_start = time.time()
            # Sample actions
            
            one_hot_actions = []
            for i in range(args.num_agents):
                one_hot_action = np.zeros(env.action_space[0].n)
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic,recurrent_c_states, recurrent_c_states_critic = actor_critic[i].act(share_obs[i], obs[i], recurrent_hidden_statess[i], recurrent_hidden_statess_critic[i], recurrent_c_statess[i], recurrent_c_statess_critic[i], masks[i], deterministic=True)
                recurrent_hidden_statess[i].copy_(recurrent_hidden_states)
                recurrent_hidden_statess_critic[i].copy_(recurrent_hidden_states_critic) 
                recurrent_c_statess[i].copy_(recurrent_c_states)
                recurrent_c_statess_critic[i].copy_(recurrent_c_states_critic)              
                one_hot_action[action] = 1
                one_hot_actions.append(one_hot_action)
             
            # Obser reward and next obs
            state, reward, done, infos = env.step(one_hot_actions)
            for i in range(args.num_agents):
                print("Reward of agent%i: " %i + str(reward[i]))
            state = np.array([state])
                        
            for i in range(args.num_agents):
                share_obs[i].copy_(torch.tensor(state.reshape(1, -1),dtype=torch.float32))
                obs[i].copy_(torch.tensor(state[:,i,:],dtype=torch.float32))

            if args.save_gifs:
                #env.render('human')
                frames.append(env.render('rgb_array')[0])

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < args.ifi:
                time.sleep(args.ifi - elapsed)
                
        
        for i in range(args.num_agents):
            if cover_landmark(env.agents[i]):
                cover += 1

        
        if args.save_gifs:
            gif_num = 0
            while os.path.exists(str(gif_dir) + '/%i_%i.gif' % (gif_num, episode)):
                gif_num += 1
            imageio.mimsave(str(gif_dir) + '/%i_%i.gif' % (gif_num, episode),
                            frames, duration=args.ifi)
        
    print("Eval done. Cover rate is %i/%i." % (cover, args.eval_episodes*args.num_agents))
 

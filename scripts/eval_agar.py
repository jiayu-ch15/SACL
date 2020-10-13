import copy
import glob
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from envs import AgarEnv
from config import get_config
import shutil
    
def main():
    args = get_config()
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    
    run_dir = Path(args.model_dir)/ ("run" + str(args.seed)) / 'eval'
    if os.path.exists(run_dir): 
        shutil.rmtree(run_dir)
        os.mkdir(run_dir)
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    logger = SummaryWriter(str(log_dir))
    gifs_dir = run_dir / 'gifs'	
    os.makedirs(str(gifs_dir))
	
    # env	
    if args.env_name == "Agar":	
        env = AgarEnv(args)		
    else:	
        print("Can not support the " + args.env_name + "environment." )	
        raise NotImplementedError	

    #Policy network    	    
    actor_critic = []	    
    for i in range(args.num_agents):
        ac = torch.load(str(args.model_dir) + 'run' + str(args.seed) + "/models/agent%i_model" % i + ".pt")['model'].to(device)
        actor_critic.append(ac)
        
    eval_rewards = []
    frames = []
        
    for episode in range(args.eval_episodes):  
        print("Episode %i of %i" % (episode, args.eval_episodes))
        state, _ = env.reset()	
        state = np.array([state])	
        
        share_obs = []	
        obs = []	
        recurrent_hidden_statess = []	
        recurrent_hidden_statess_critic = []	
        recurrent_c_statess = []	
        recurrent_c_statess_critic = []	
        masks = []	
        policy_reward = 0	

        # rollout	
        for i in range(args.num_agents):		
            share_obs.append((torch.tensor(state.reshape(1, -1),dtype=torch.float32)).to(device))	
            obs.append((torch.tensor(state[:,i,:],dtype=torch.float32)).to(device))		
            recurrent_hidden_statess.append(torch.zeros(1, actor_critic[i].recurrent_hidden_size).to(device))	
            recurrent_hidden_statess_critic.append(torch.zeros(1, actor_critic[i].recurrent_hidden_size).to(device))	
            recurrent_c_statess.append(torch.zeros(1, actor_critic[i].recurrent_hidden_size).to(device))	
            recurrent_c_statess_critic.append(torch.zeros(1, actor_critic[i].recurrent_hidden_size).to(device))	
            masks.append(torch.ones(1,1).to(device))	
            
        frames_dir = str(gifs_dir) + '/episode%i/'%episode + 'frames/'
        for step in range(args.episode_length):	   
            print("step %i of %i" % (step, args.episode_length))	
            # Sample actions
            
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
                
            if args.save_gifs:
                frame = env.render(0, mode='rgb_array', name= str(gifs_dir) + '/' + str(args.env_name) + '_trajectory')
                frames.append(frame)	

            actions_env = []
            for i in range(args.num_agents):	
                with torch.no_grad():	
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic,recurrent_c_states, recurrent_c_states_critic = actor_critic[i].act(i, share_obs[i], obs[i], recurrent_hidden_statess[i], recurrent_hidden_statess_critic[i], recurrent_c_statess[i], recurrent_c_statess_critic[i], masks[i])	
                recurrent_hidden_statess[i].copy_(recurrent_hidden_states)	
                recurrent_hidden_statess_critic[i].copy_(recurrent_hidden_states_critic) 	
                recurrent_c_statess[i].copy_(recurrent_c_states)	
                recurrent_c_statess_critic[i].copy_(recurrent_c_states_critic) 	
                actions_env.append(action[0].cpu().tolist())	

            # Obser reward and next obs	
            print(actions_env)
            actions_env = np.array(actions_env)
            state, reward, done, infos, _ = env.step(actions_env)	

            for i in range(args.num_agents):	
                print("Reward of agent%i: " %i + str(reward[i]))	
                policy_reward += reward[i]	
            state = np.array([state])	

            for i in range(args.num_agents):		
                share_obs[i].copy_(torch.tensor(state.reshape(1, -1),dtype=torch.float32))	
                obs[i].copy_(torch.tensor(state[:,i,:],dtype=torch.float32))	
                    
        eval_rewards.append(policy_reward)
        #if args.save_gifs:
        #    utility_funcs.make_gif_from_image_dir(str(gifs_dir) + '/episode%i/'%episode, frames_dir, gif_name=args.env_name + '_trajectory')
        
        if args.env_name == "Agar":            
            for agent_id in range(args.num_agents):
                collective_return = []
                split = []
                hunt = []
                attack = []
                cooperate = []                   
                if 'collective_return' in infos[agent_id].keys():
                    collective_return.append(infos[agent_id]['collective_return']) 
                if 'behavior' in infos[agent_id].keys():
                    split.append(infos[agent_id]['behavior'][0])
                    hunt.append(infos[agent_id]['behavior'][1])
                    attack.append(infos[agent_id]['behavior'][2])
                    cooperate.append(infos[agent_id]['behavior'][3])                                                     

                logger.add_scalars('agent%i/collective_return' % agent_id,
                                {'collective_return': np.mean(collective_return)},
                                episode)
                logger.add_scalars('agent%i/split' % agent_id,
                                    {'split': np.mean(split)},
                                    episode)
                logger.add_scalars('agent%i/hunt' % agent_id,
                                    {'hunt': np.mean(hunt)},
                                    episode)
                logger.add_scalars('agent%i/attack' % agent_id,
                                    {'attack': np.mean(attack)},
                                    episode)
                logger.add_scalars('agent%i/cooperate' % agent_id,
                                    {'cooperate': np.mean(cooperate)},
                                    episode)
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()   
    r = np.mean(np.array(eval_rewards))
    print("Mean reward is %i" % r)
    env.close()
    
if __name__ == "__main__":
    main()
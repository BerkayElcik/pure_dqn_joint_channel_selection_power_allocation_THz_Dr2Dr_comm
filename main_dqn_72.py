#import gym
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

import numpy as np
from dqn_agent_72 import DQNAgent
from utils import plot_learning_curve
#from gym import wrappers
from gymnasium import wrappers
from terahertz_drone_environment import thz_drone_env

if __name__ == '__main__':
    #env = make_env('CUSTOM_ENV(not_ready)')
    #env = gym.make('CUSTOM_ENV(not_ready)')
    env=thz_drone_env(n_channels=1217, P_T=1, freq_of_movement=0.1)
    env = FlattenObservation(env)


    best_capacity = -np.inf
    load_checkpoint = False
    n_games = 22

    """
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')
    """

    num_actions = np.prod(env.action_space.nvec)


    """
    #print(np.random.choice(env.action_space))
    print(env.action_space.sample())
    print(env.observation_space.shape)
    print(env.action_space.shape)
    print(env.observation_space.shape[0])
    print(env.action_space.nvec)
    print(num_actions)
    print(env.action_space.sample())
    print(env.observation_space.sample())
    print(env.observation_space.sample().size)
    """

    agent = DQNAgent(input_dims=env.observation_space.shape[0],
                     n_actions=num_actions,
                     mem_size=2000,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='THz_channel_selection')



    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation, info = env.reset()
        """
        print("obsmain")
        print(observation)
        print(len(observation))
        print(info)
        """

        Score = 0
        #while not done:
        for i in range(10000):
            action, action_index = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            print("---------------------OBSERVATION---------------------")
            print(observation)
            print("---------------------ACTION---------------------")
            print(action)
            print("---------------------NEXT OBSERVATION---------------------")
            print(observation_)
            print("---------------------REWARD---------------------")
            print(reward)
            print("---------------------INFO---------------------")
            print(info)
            """
            print("obs_main")
            print(observation_)
            print(len(observation_))
            """
            Score += reward

            print("---------------------SCORE---------------------")
            print(Score)

            if not load_checkpoint:
                agent.store_transition(observation, action, action_index,
                                     reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(Score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'Capacity: ', info["capacity"], 'Score: ', Score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps, "no of channels:", info['no_of_channels'])

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)

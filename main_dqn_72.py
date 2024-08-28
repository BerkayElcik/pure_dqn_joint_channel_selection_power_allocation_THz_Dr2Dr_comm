#import gym
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

import numpy as np
from dqn_agent_72 import DQNAgent
from utils import plot_learning_curve
#from gym import wrappers
from gymnasium import wrappers
from terahertz_drone_environment import thz_drone_env
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #env = make_env('CUSTOM_ENV(not_ready)')
    #env = gym.make('CUSTOM_ENV(not_ready)')
    env=thz_drone_env(n_channels=50, P_T=1, freq_of_movement=0.1)

    env = FlattenObservation(env)


    best_score = -np.inf
    load_checkpoint = False
    n_games = 55

    """
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')
    """

    num_actions = np.prod(env.action_space.nvec)



    #print(np.random.choice(env.action_space))
    #print(env.action_space.sample())
    #print(env.observation_space.shape)
    #print(env.action_space.shape)
    print(env.observation_space.shape[0])
    print(env.observation_space.shape)
    print(env.observation_space)
    print(env.action_space.nvec)
    print(num_actions)
    #print(env.action_space.sample())
    #print(env.observation_space.sample())
    #print(env.observation_space.sample().size)


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
    Capacity_graph = []

    distance_1_graph_scores, distance_2_graph_scores, distance_3_graph_scores, distance_4_graph_scores, distance_5_graph_scores, \
    distance_6_graph_scores, distance_7_graph_scores, distance_8_graph_scores, distance_9_graph_scores, distance_10_graph_scores, \
    distance_11_graph_scores = [], [], [], [], [], [], [], [], [], [], []

    distance_1_graph_capacity, distance_2_graph_capacity, distance_3_graph_capacity, distance_4_graph_capacity, distance_5_graph_capacity, \
    distance_6_graph_capacity, distance_7_graph_capacity, distance_8_graph_capacity, distance_9_graph_capacity, distance_10_graph_capacity, \
    distance_11_graph_capacity = [], [], [], [], [], [], [], [], [], [], []


    print("Started Training")
    n_activity=30000
    num_training_steps=n_games*n_activity
    progress_bar = tqdm(range(num_training_steps))

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
        for i in range(n_activity):
            action, action_index = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)


            """
            print("---------------------OBSERVATION---------------------")
            print(observation)
            print(observation.shape)
            """

            """
            print("---------------------ACTION---------------------")
            print(action)
            """

            """
            print("---------------------NEXT OBSERVATION---------------------")
            print(observation_)
            """


            print("---------------------REWARD---------------------")
            print(reward)


            """
            print("---------------------INFO---------------------")
            print(info)
            """

            Score += reward
            """
            print("---------------------SCORE---------------------")
            print(Score)
            """



            if not load_checkpoint:
                agent.store_transition(observation, action, action_index,
                                     reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
            progress_bar.update(1)
        scores.append(Score)
        steps_array.append(n_steps)
        Capacity_graph.append(info["capacity"])

        if observation[50] == 1:
            distance_1_graph_scores.append(Score)
            distance_1_graph_capacity.append(info["capacity"])
        elif observation[50] == 2:
            distance_2_graph_scores.append(Score)
            distance_2_graph_capacity.append(info["capacity"])
        elif observation[50] == 3:
            distance_3_graph_scores.append(Score)
            distance_3_graph_capacity.append(info["capacity"])
        elif observation[50] == 4:
            distance_4_graph_scores.append(Score)
            distance_4_graph_capacity.append(info["capacity"])
        elif observation[50] == 5:
            distance_5_graph_scores.append(Score)
            distance_5_graph_capacity.append(info["capacity"])
        elif observation[50] == 6:
            distance_6_graph_scores.append(Score)
            distance_6_graph_capacity.append(info["capacity"])
        elif observation[50] == 7:
            distance_7_graph_scores.append(Score)
            distance_7_graph_capacity.append(info["capacity"])
        elif observation[50] == 8:
            distance_8_graph_scores.append(Score)
            distance_8_graph_capacity.append(info["capacity"])
        elif observation[50] == 9:
            distance_9_graph_scores.append(Score)
            distance_9_graph_capacity.append(info["capacity"])
        elif observation[50] == 10:
            distance_10_graph_scores.append(Score)
            distance_10_graph_capacity.append(info["capacity"])
        elif observation[50] == 11:
            distance_11_graph_scores.append(Score)
            distance_11_graph_capacity.append(info["capacity"])

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


    figure_file = 'plots/' + "Capacity_over_time" + '.png'
    plot_learning_curve(steps_array, Capacity_graph, eps_history, figure_file)

    import matplotlib.pyplot as plt
    import numpy as np

    # Plot for distance_1_graph_capacity
    figure_file = 'plots/' + "distance_1_graph_capacity" + '.png'
    abscissa_len = len(distance_1_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_1_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_1_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_1_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_1_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_1_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_1_graph_scores
    figure_file = 'plots/' + "distance_1_graph_scores" + '.png'
    abscissa_len = len(distance_1_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_1_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_1_graph_scores_moving_average
    figure_file = 'plots/' + "distance_1_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_1_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_1_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_2_graph_capacity
    figure_file = 'plots/' + "distance_2_graph_capacity" + '.png'
    abscissa_len = len(distance_2_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_2_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_2_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_2_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_2_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_2_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_2_graph_scores
    figure_file = 'plots/' + "distance_2_graph_scores" + '.png'
    abscissa_len = len(distance_2_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_2_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_2_graph_scores_moving_average
    figure_file = 'plots/' + "distance_2_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_2_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_2_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_3_graph_capacity
    figure_file = 'plots/' + "distance_3_graph_capacity" + '.png'
    abscissa_len = len(distance_3_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_3_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_3_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_3_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_3_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_3_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_3_graph_scores
    figure_file = 'plots/' + "distance_3_graph_scores" + '.png'
    abscissa_len = len(distance_3_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_3_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_3_graph_scores_moving_average
    figure_file = 'plots/' + "distance_3_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_3_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_3_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_4_graph_capacity
    figure_file = 'plots/' + "distance_4_graph_capacity" + '.png'
    abscissa_len = len(distance_4_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_4_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_4_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_4_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_4_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_4_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_4_graph_scores
    figure_file = 'plots/' + "distance_4_graph_scores" + '.png'
    abscissa_len = len(distance_4_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_4_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_4_graph_scores_moving_average
    figure_file = 'plots/' + "distance_4_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_4_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_4_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_5_graph_capacity
    figure_file = 'plots/' + "distance_5_graph_capacity" + '.png'
    abscissa_len = len(distance_5_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_5_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_5_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_5_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_5_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_5_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_5_graph_scores
    figure_file = 'plots/' + "distance_5_graph_scores" + '.png'
    abscissa_len = len(distance_5_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_5_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_5_graph_scores_moving_average
    figure_file = 'plots/' + "distance_5_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_5_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_5_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_6_graph_capacity
    figure_file = 'plots/' + "distance_6_graph_capacity" + '.png'
    abscissa_len = len(distance_6_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_6_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_6_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_6_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_6_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_6_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_6_graph_scores
    figure_file = 'plots/' + "distance_6_graph_scores" + '.png'
    abscissa_len = len(distance_6_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_6_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_6_graph_scores_moving_average
    figure_file = 'plots/' + "distance_6_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_6_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_6_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_7_graph_capacity
    figure_file = 'plots/' + "distance_7_graph_capacity" + '.png'
    abscissa_len = len(distance_7_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_7_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_7_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_7_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_7_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_7_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_7_graph_scores
    figure_file = 'plots/' + "distance_7_graph_scores" + '.png'
    abscissa_len = len(distance_7_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_7_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_7_graph_scores_moving_average
    figure_file = 'plots/' + "distance_7_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_7_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_7_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_8_graph_capacity
    figure_file = 'plots/' + "distance_8_graph_capacity" + '.png'
    abscissa_len = len(distance_8_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_8_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_8_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_8_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_8_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_8_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_8_graph_scores
    figure_file = 'plots/' + "distance_8_graph_scores" + '.png'
    abscissa_len = len(distance_8_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_8_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_8_graph_scores_moving_average
    figure_file = 'plots/' + "distance_8_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_8_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_8_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_9_graph_capacity
    figure_file = 'plots/' + "distance_9_graph_capacity" + '.png'
    abscissa_len = len(distance_9_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_9_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_9_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_9_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_9_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_9_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_9_graph_scores
    figure_file = 'plots/' + "distance_9_graph_scores" + '.png'
    abscissa_len = len(distance_9_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_9_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_9_graph_scores_moving_average
    figure_file = 'plots/' + "distance_9_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_9_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_9_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_10_graph_capacity
    figure_file = 'plots/' + "distance_10_graph_capacity" + '.png'
    abscissa_len = len(distance_10_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_10_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_10_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_10_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_10_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_10_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_10_graph_scores
    figure_file = 'plots/' + "distance_10_graph_scores" + '.png'
    abscissa_len = len(distance_10_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_10_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_10_graph_scores_moving_average
    figure_file = 'plots/' + "distance_10_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_10_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_10_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Repeat for distance_11_graph_capacity
    figure_file = 'plots/' + "distance_11_graph_capacity" + '.png'
    abscissa_len = len(distance_11_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_11_graph_capacity, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_11_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_11_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_11_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_11_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_11_graph_scores
    figure_file = 'plots/' + "distance_11_graph_scores" + '.png'
    abscissa_len = len(distance_11_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_11_graph_scores, color="C0")
    plt.savefig(figure_file)
    plt.show()

    # Plot for distance_11_graph_scores_moving_average
    figure_file = 'plots/' + "distance_11_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_11_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_11_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C0")
    plt.savefig(figure_file)
    plt.show()

    #CALCULATE MAX VALUE
    max_distance_1_score = np.max(distance_1_graph_scores)
    max_distance_1_capacity = np.max(distance_1_graph_capacity)

    max_distance_2_score = np.max(distance_2_graph_scores)
    max_distance_2_capacity = np.max(distance_2_graph_capacity)

    max_distance_3_score = np.max(distance_3_graph_scores)
    max_distance_3_capacity = np.max(distance_3_graph_capacity)

    max_distance_4_score = np.max(distance_4_graph_scores)
    max_distance_4_capacity = np.max(distance_4_graph_capacity)

    max_distance_5_score = np.max(distance_5_graph_scores)
    max_distance_5_capacity = np.max(distance_5_graph_capacity)

    max_distance_6_score = np.max(distance_6_graph_scores)
    max_distance_6_capacity = np.max(distance_6_graph_capacity)

    max_distance_7_score = np.max(distance_7_graph_scores)
    max_distance_7_capacity = np.max(distance_7_graph_capacity)

    max_distance_8_score = np.max(distance_8_graph_scores)
    max_distance_8_capacity = np.max(distance_8_graph_capacity)

    max_distance_9_score = np.max(distance_9_graph_scores)
    max_distance_9_capacity = np.max(distance_9_graph_capacity)

    max_distance_10_score = np.max(distance_10_graph_scores)
    max_distance_10_capacity = np.max(distance_10_graph_capacity)

    max_distance_11_score = np.max(distance_11_graph_scores)
    max_distance_11_capacity = np.max(distance_11_graph_capacity)

    distance_to_score_graph=[max_distance_1_score, max_distance_2_score, max_distance_3_score, max_distance_4_score, max_distance_5_score, max_distance_6_score, max_distance_7_score, max_distance_8_score, max_distance_9_score, max_distance_10_score, max_distance_11_score]

    distance_to_capacity_graph = [max_distance_1_capacity, max_distance_2_capacity, max_distance_3_capacity, max_distance_4_capacity,
                               max_distance_5_capacity, max_distance_6_capacity, max_distance_7_capacity, max_distance_8_capacity,
                               max_distance_9_capacity, max_distance_10_capacity, max_distance_11_capacity]



    figure_file = 'plots/' + "distance_to_score_graph" + '.png'
    abscissa_len_distance_score = len(distance_to_score_graph)
    abscissa = np.arange(abscissa_len_distance_score)
    plt.scatter(abscissa, distance_to_score_graph, color="C0")
    plt.savefig(figure_file)
    plt.show()



    figure_file = 'plots/' + "distance_to_capacity_graph" + '.png'
    abscissa_len_distance_capacity = len(distance_to_capacity_graph)
    abscissa = np.arange(abscissa_len_distance_capacity)
    plt.scatter(abscissa, distance_to_score_graph, color="C0")
    plt.savefig(figure_file)
    plt.show()





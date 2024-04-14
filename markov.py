import gymnasium as gym
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.grid_search import GridSearch
from bettermdptools.algorithms.rl import RL
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_plot_file(name):
    os.makedirs('results', exist_ok=True)
    filename = f'results/{name}.png'
    plt.savefig(filename)
    plt.clf()

def v_iters_plot(data, title):
    df = pd.DataFrame(data=data)
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, legend=None).set_title(title)
    save_plot_file(title)

def get_policy_map(pi, val_max, actions, map_size):
    """Map the best learned action to arrows."""
    #convert pi to numpy array
    best_action = np.zeros(val_max.shape[0], dtype=np.int32)
    for idx, val in enumerate(val_max):
        best_action[idx] = pi[idx]
    policy_map = np.empty(best_action.flatten().shape, dtype=str)
    for idx, val in enumerate(best_action.flatten()):
        policy_map[idx] = actions[val]
    policy_map = policy_map.reshape(map_size[0], map_size[1])
    val_max = val_max.reshape(map_size[0], map_size[1])
    return val_max, policy_map

def plot_policy(val_max, directions, map_size, title, fontsize="small"):
    """Plot the policy learned."""
    sns.heatmap(
        val_max,
        annot=directions,
        fmt="",
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": fontsize},
    ).set(title=title)
    img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"
    save_plot_file(title)

def values_heat_map(data, title, size):
    data = np.around(np.array(data).reshape(size), 2)
    df = pd.DataFrame(data=data)
    sns.heatmap(df, annot=True).set_title(title)
    save_plot_file(title)

def run_blackjack(seed=44):

    gamma=[0.0]
    n_iters=[10,20,30]
    theta=[.001,.0001]
    base_env = gym.make('Blackjack-v1', render_mode='rgb_array')
    base_env.reset(seed=seed)
    blackjack = BlackjackWrapper(base_env)
    # GridSearch.vi_grid_search(blackjack, gamma, n_iters, theta)
    # GridSearch.pi_grid_search(blackjack, gamma, n_iters, theta)

    gamma = 1.0
    iterations = 1000
    theta = .0001

    # run VI
    V, V_track, pi = Planner(blackjack.P).value_iteration(gamma=gamma, n_iters=iterations, theta=theta)
    episode_rewards = TestEnv.test_env(env=blackjack, render=False, user_input=False, pi=pi)
    print(f'Policy mean score VI - Blackjack: {np.mean(episode_rewards)}')


    max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), 'b')
    v_iters_plot(max_value_per_iter, "Blackjack Mean Value v Iterations - VI")

    blackjack_actions = {0: "S", 1: "H"}
    blackjack_map_size=(29, 10)

    val_max, policy_map = get_policy_map(pi, V, blackjack_actions, blackjack_map_size)

    title="Blackjack Policy Map - VI"
    plot_policy(val_max, policy_map, blackjack_map_size, title)

    gamma = 1.0
    iterations = 1000
    theta = .0001

    #run PI
    V_pi, V_track_pi, pi_pi = Planner(blackjack.P).policy_iteration(gamma=gamma, n_iters=iterations, theta=theta)
    episode_rewards_pi = TestEnv.test_env(env=blackjack, render=False, user_input=False, pi=pi_pi)
    print(f'Policy mean score PI - Blackjack: {np.mean(episode_rewards_pi)}')

    max_value_per_iter_pi = np.trim_zeros(np.mean(V_track_pi, axis=1), 'b')
    v_iters_plot(max_value_per_iter_pi, "Blackjack Mean Value v Iterations - PI")

    val_max, policy_map = get_policy_map(pi_pi, V_pi, blackjack_actions, blackjack_map_size)

    title="Blackjack Policy Map - PI"
    plot_policy(val_max, policy_map, blackjack_map_size, title)

def run_blackjack_q(seed=44):

    base_env = gym.make('Blackjack-v1', render_mode='rgb_array')
    base_env.reset(seed=seed)
    blackjack = BlackjackWrapper(base_env)

    gammas=[.1,.5,.9]
    epsilon_decays = [0.01,0.1,0.9]
    iters = [5000, 10000]
    # GridSearch.q_learning_grid_search(blackjack, gammas, epsilon_decays, iters)    

    gamma = 0.5
    iterations = 5000
    epsilon_decay = 0.9

    # Q-learning
    Q, V, pi, Q_track, pi_track = RL(blackjack).q_learning(gamma=gamma, epsilon_decay_ratio=epsilon_decay, n_episodes=iterations)

    #test policy
    episode_rewards = TestEnv.test_env(env=blackjack, n_iters=100, pi=pi)
    print(f'Policy mean score Q Learning - Blackjack: {np.mean(episode_rewards)}')

    # max_value_per_iter_q = np.trim_zeros(np.mean(Q_track, axis=1), 'b')
    # v_iters_plot(max_value_per_iter_q, "Blackjack Mean Value v Iterations - Q Learning")
    # v_iters_plot(episode_rewards, "Blackjack Rewards v Episodes - Q Learning")

    blackjack_actions = {0: "S", 1: "H"}
    blackjack_map_size=(29, 10)

    val_max, policy_map = get_policy_map(pi, V, blackjack_actions, blackjack_map_size)

    title="Blackjack Policy Map - Q Learning"
    plot_policy(val_max, policy_map, blackjack_map_size, title)

def run_taxi(seed=44):

    # make gym environment 
    taxi = gym.make('Taxi-v3', render_mode=None)
    taxi.reset(seed=seed)

    # gamma=[.99]
    # epsilon_decay = [.9]
    # iters = [500, 5000, 50000]
    # GridSearch.q_learning_grid_search(taxi, gamma, epsilon_decay, iters)    

    iterations = 1000
    gamma = 0.99
    theta = 0.001

    V, V_track, pi = Planner(taxi.P).value_iteration(n_iters=iterations, gamma=gamma, theta=theta)

    episode_rewards = TestEnv.test_env(env=taxi, render=False, user_input=False, pi=pi)
    print(f'Policy mean score VI - Taxi: {np.mean(episode_rewards)}')

    max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), 'b')
    v_iters_plot(max_value_per_iter, "Taxi Mean Value v Iterations - VI")

    # size=(20, 25)
    # values_heat_map(V, "Taxi Value Iteration State Values - VI", size)      
    taxi_actions = {0: "←", 1: "↓", 2: "→", 3: "↑", 4: "P", 5: "D"}
    taxi_map_size=(20, 25)
    title="Taxi Policy Map - VI"
    val_max, policy_map = get_policy_map(pi, V, taxi_actions, taxi_map_size)
    plot_policy(val_max, policy_map, taxi_map_size, title, "small")


    iterations = 5000
    gamma = 0.9
    theta = 0.0001

    V_pi, V_track_pi, pi_pi = Planner(taxi.P).policy_iteration(n_iters=iterations, gamma=gamma, theta=theta)

    episode_rewards_pi = TestEnv.test_env(env=taxi, render=False, user_input=False, pi=pi_pi)
    print(f'Policy mean score PI - Taxi: {np.mean(episode_rewards_pi)}')


    max_value_per_iter_pi = np.trim_zeros(np.mean(V_track_pi, axis=1), 'b')
    v_iters_plot(max_value_per_iter_pi, "Taxi Mean Value v Iterations - PI")    

    title="Taxi Policy Map - PI"
    val_max, policy_map = get_policy_map(pi_pi, V_pi, taxi_actions, taxi_map_size)
    plot_policy(val_max, policy_map, taxi_map_size, title, "small")

    # episodes = range(len(episode_rewards))
    # plt.plot(episodes, episode_rewards, label="Value Iteration")
    # plt.plot(episodes, episode_rewards_pi, label="Policy Iteration")
    # plt.xlabel("Episodes")
    # plt.ylabel("Average Reward")
    # plt.title("Policy Convergence Comparison (Taxi MDP)")
    # plt.legend()
    # plt.show()

    print(episode_rewards)

def run_taxi_q(seed=44):

    taxi = gym.make('Taxi-v3', render_mode=None)
    taxi.reset(seed=seed)

    gammas=[.9,.99]
    epsilon_decays = [.01,.1]
    iters = [5000,10000]
    # GridSearch.q_learning_grid_search(taxi, gammas, epsilon_decays, iters)    

    iterations = 10000
    gamma = 0.9
    epsilon_decay = 0.1

    # Q-learning
    Q, V_q, pi_q, Q_track, pi_track = RL(taxi).q_learning(gamma=gamma, n_episodes=iterations, epsilon_decay_ratio=epsilon_decay)

    #test policy
    test_scores = TestEnv.test_env(env=taxi, n_iters=iterations, render=False, pi=pi_q, user_input=False)
    print(f'Policy mean score Q Learning - Taxi: {np.mean(test_scores)}')
    # max_value_per_iter_q = np.trim_zeros(np.mean(Q_track, axis=1), 'b')

    taxi_actions = {0: "←", 1: "↓", 2: "→", 3: "↑", 4: "P", 5: "D"}
    taxi_map_size=(20, 25)
    val_max, policy_map = get_policy_map(pi_q, V_q, taxi_actions, taxi_map_size)

    title="Taxi Policy Map - Q Learning"
    plot_policy(val_max, policy_map, taxi_map_size, title)


run_blackjack(44)
run_blackjack_q(44)
run_taxi(44)
run_taxi_q(44)
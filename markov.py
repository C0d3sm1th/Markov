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

def plot_policy(val_max, directions, map_size, title):
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
        annot_kws={"fontsize": "xx-large"},
    ).set(title=title)
    img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"
    save_plot_file(title)

gamma=[.8,.9,1.0]
n_iters=[10,20,50]
theta=[.1,.01,.001]
base_env = gym.make('Blackjack-v1', render_mode='rgb_array')
base_env.reset(seed=44)
blackjack = BlackjackWrapper(base_env)
# GridSearch.vi_grid_search(blackjack, gamma, n_iters, theta)
# GridSearch.pi_grid_search(blackjack, gamma, n_iters, theta)

gamma = 1.0
iterations = 50
theta=0.001

# run VI
# V, V_track, pi = Planner(blackjack.P).value_iteration()
V, V_track, pi = Planner(blackjack.P).value_iteration(gamma=gamma, n_iters=iterations, theta=theta)
max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), 'b')
v_iters_plot(max_value_per_iter, "Blackjack Mean Value v Iterations - VI")

blackjack_actions = {0: "S", 1: "H"}
blackjack_map_size=(29, 10)

val_max, policy_map = get_policy_map(pi, V, blackjack_actions, blackjack_map_size)

title="Blackjack Policy Map - VI"
plot_policy(val_max, policy_map, blackjack_map_size, title)

gamma=1.0
iterations = 20
theta=0.01

#test policy
V_pi, V_track_pi, pi_pi = Planner(blackjack.P).policy_iteration(gamma=gamma, n_iters=iterations, theta=theta)
max_value_per_iter_pi = np.trim_zeros(np.mean(V_track_pi, axis=1), 'b')
v_iters_plot(max_value_per_iter_pi, "Blackjack Mean Value v Iterations - PI")

val_max, policy_map = get_policy_map(pi_pi, V_pi, blackjack_actions, blackjack_map_size)

title="Blackjack Policy Map - PI"
plot_policy(val_max, policy_map, blackjack_map_size, title)


# Q-learning
Q, V, pi, Q_track, pi_track = RL(blackjack).q_learning(gamma=gamma)

#test policy
test_scores = TestEnv.test_env(env=blackjack, n_iters=iterations, render=False, pi=pi, user_input=False)
print(f'Test policy mean score Q Learning: {np.mean(test_scores)}')

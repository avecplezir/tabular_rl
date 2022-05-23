import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.helper_functions_multitask import create_policy_direction_arrays
import numpy as np
import copy

def plot_gridworld(model, value_function=None, policy=None, state_counts=None, title=None, path=None,
                   start_states=[None], goal_states=[None]):
    """
    Plots the grid world solution.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    value_function : numpy array of shape (N, 1)
        Value function of the environment where N is the number
        of states in the environment.

    policy : numpy array of shape (N, 1)
        Optimal policy of the environment.

    title : string
        Title of the plot. Defaults to None.

    path : string
        Path to save image. Defaults to None.
    """

    if value_function is not None and state_counts is not None:
        raise Exception("Must supple either value function or state_counts, not both!")

    fig, axes = plt.subplots(len(start_states), len(goal_states))

    if len(start_states)*len(goal_states)==1:
        axes = np.array([[axes]])
    elif len(goal_states)==1:
        axes = axes[:, None]
    elif len(start_states) == 1:
        axes = axes[None, :]

    # print('axes', axes)

    for i, start_state in enumerate(start_states):
        for j, goal_state in enumerate(goal_states):
            # print('i, j', i, j)
            # add features to grid world
            if value_function is not None:
                add_value_function(model, copy.deepcopy(value_function), "Value function", axes[i,j])
            # elif state_counts is not None:
            #     add_value_function(model, state_counts, "State counts")
            # elif value_function is None and state_counts is None:
            #     add_value_function(model, value_function, "Value function")

            add_patches(model, axes[i,j], start_state, goal_state)
            # print('patches added')
            add_policy(model, policy, axes[i,j], start_state=start_state, goal_state=goal_state)
            # print('policy added')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=3)
    if title is not None:
        plt.title(title, fontdict=None, loc='center')
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_gridworld_subtasks(model, value_function=None, policy=None, state_counts=None, title=None, path=None,
                   start_states=[None], goal_states=[None], subgoal_states=[None]):
    """
    Plots the grid world solution.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    value_function : numpy array of shape (N, 1)
        Value function of the environment where N is the number
        of states in the environment.

    policy : numpy array of shape (N, 1)
        Optimal policy of the environment.

    title : string
        Title of the plot. Defaults to None.

    path : string
        Path to save image. Defaults to None.
    """

    if value_function is not None and state_counts is not None:
        raise Exception("Must supple either value function or state_counts, not both!")

    fig, axes = plt.subplots(len(start_states))

    if len(start_states)==1:
        axes = np.array([axes])

    for i, (start_state, goal_state, subgoal_state) in enumerate(zip(start_states, goal_states, subgoal_states)):
        # add features to grid world
        if value_function is not None:
            add_value_function(model, copy.deepcopy(value_function), "Value function", axes[i])
        # elif state_counts is not None:
        #     add_value_function(model, state_counts, "State counts")
        # elif value_function is None and state_counts is None:
        #     add_value_function(model, value_function, "Value function")

        add_patches(model, axes[i], start_state, goal_state, subgoal_state)
        # print('start_state', start_state.shape, subgoal_state.shape, goal_state.shape)
        # for subgoal in subgoal_state:
        if subgoal_state[0] is not None:
            add_policy(model, policy, axes[i], start_state=start_state, goal_state=subgoal_state[0])
            add_policy(model, policy, axes[i], start_state=subgoal_state[0], goal_state=goal_state)
        else:
            add_policy(model, policy, axes[i], start_state=start_state, goal_state=goal_state)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=3)
    if title is not None:
        plt.title(title, fontdict=None, loc='center')
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()

def add_value_function(model, value_function, name, ax):

    if value_function is not None:
        # colobar max and min
        vmin = np.min(value_function)
        vmax = np.max(value_function)
        # reshape and set obstructed states to low value
        val = value_function[:-1, 0].reshape(model.num_rows, model.num_cols)
        if model.obs_states is not None:
            index = model.obs_states
            val[index[:, 0], index[:, 1]] = -100
        ax.imshow(val, vmin=vmin, vmax=vmax, zorder=0)
        # ax.colorbar(label=name)
    else:
        val = np.zeros((model.num_rows, model.num_cols))
        ax.imshow(val, zorder=0)
        ax.yticks(np.arange(-0.5, model.num_rows+0.5, step=1))
        ax.xticks(np.arange(-0.5, model.num_cols+0.5, step=1))
        ax.grid()
        ax.colorbar(label=name)

def add_patches(model, ax, start_state, goal_state, subgoal_states=[None]):

    start = patches.Circle(tuple(np.flip(start_state)), 0.2, linewidth=1,
                           edgecolor='b', facecolor='b', zorder=1, label="Start")
    ax.add_patch(start)

    end = patches.RegularPolygon(tuple(np.flip(goal_state)), numVertices=5,
                                 radius=0.25, orientation=np.pi, edgecolor='g', zorder=1,
                                 facecolor='g', label="Goal")
    ax.add_patch(end)

    # obstructed states patches
    if model.obs_states is not None:
        for i in range(model.obs_states.shape[0]):
            obstructed = patches.Rectangle(tuple(np.flip(model.obs_states[i, :]) - 0.35), 0.7, 0.7,
                                           linewidth=1, edgecolor='orange', facecolor='orange', zorder=1,
                                           label="Obstructed" if i == 0 else None)
            ax.add_patch(obstructed)

    if model.bad_states is not None:
        for i in range(model.bad_states.shape[0]):
            bad = patches.Wedge(tuple(np.flip(model.bad_states[i, :])), 0.2, 40, -40,
                                linewidth=1, edgecolor='r', facecolor='r', zorder=1,
                                label="Bad state" if i == 0 else None)
            ax.add_patch(bad)

    # if model.restart_states is not None:
    #     for i in range(model.restart_states.shape[0]):
    #         restart = patches.Wedge(tuple(np.flip(model.restart_states[i, :])), 0.2, 40, -40,
    #                                 linewidth=1, edgecolor='y', facecolor='y', zorder=1,
    #                                 label="Restart state" if i == 0 else None)
    #         ax.add_patch(restart)

    for i in range(len(subgoal_states)):
        if subgoal_states[i] is not None:
            subgoal = patches.Wedge(tuple(np.flip(subgoal_states[i, :])), 0.2, 40, -40,
                                    linewidth=1, edgecolor='y', facecolor='y', zorder=1,
                                    label="Subgoal state" if i == 0 else None)
            ax.add_patch(subgoal)


def add_policy(model, policy, ax, start_state, goal_state):

    if policy is not None:
        # define the gridworld
        X = np.arange(0, model.num_cols, 1)
        Y = np.arange(0, model.num_rows, 1)

        # define the policy direction arrows
        U, V = create_policy_direction_arrays(model, policy, start_state=start_state, goal_state=goal_state)
        # remove the obstructions and final state arrows
        # print('goal_state', goal_state.shape)
        ra = goal_state[None, :] #model.goal_states
        U[ra[:, 0], ra[:, 1]] = np.nan
        V[ra[:, 0], ra[:, 1]] = np.nan
        if model.obs_states is not None:
            ra = model.obs_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan
        if model.restart_states is not None:
            ra = model.restart_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan

        ax.quiver(X, Y, U, V, zorder=10, label="Policy")
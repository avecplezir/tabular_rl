import copy

import numpy as np

# global stopping criteria
EPS = 0.001

def value_iteration(model, maxiter=100):
    """
    Solves the supplied environment with value iteration.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    maxiter : int
        The maximum number of iterations to perform.

    Return
    ------
    val_ : numpy array of shape (N, 1)
        Value function of the environment where N is the number
        of states in the environment.

    pi : numpy array of shape (N, 1)
        Optimal policy of the environment.
    """
    # initialize the value function and policy
    pi = np.ones((model.num_states, model.num_states, 1))
    val_ = np.zeros((model.num_states, model.num_states, 1))

    for i in range(maxiter):
        # initialize delta
        delta = 0
        # perform Bellman update for each state
        for state in range(model.num_states):
            for goal in model.goal_states_seq:
                P = copy.deepcopy(model.P)
                # P = model.P.copy()
                P[goal, :, :] = 0
                P[goal, model.num_states - 1, :] = 1
                # print('P[goal, model.num_states - 1, :]', P[goal, :, :])
                # print('P[:, :, :]', P[:, :, :])
                # store old value
                tmp = val_[state, goal].copy()
                # compute the value function
                val_[state, goal] = np.max(np.sum((model.R[state, goal] + model.gamma * val_[:, goal, :]) * P[state,:,:], 0) )
                # find maximum change in value
                delta = np.max( (delta, np.abs(tmp - val_[state, goal])) )
                # print('delta', delta)
            # stopping criteria
        if delta <= EPS * (1 - model.gamma) / model.gamma:
            print("Value iteration converged after %d iterations." %  i)
            break
    # compute the policy
    for state in range(model.num_states):
        for goal in model.goal_states_seq:
            P = copy.deepcopy(model.P)
            P[goal, :, :] = 0
            P[goal, model.num_states - 1, :] = 1
            pi[state, goal] = np.argmax(np.sum(val_[:, goal, :] * P[state,:,:],0))

    return val_, pi

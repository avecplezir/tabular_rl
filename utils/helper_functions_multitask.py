from math import floor
import numpy as np

def row_col_to_seq(row_col, num_cols):
    return row_col[:,0] * num_cols + row_col[:,1]

def seq_to_col_row(seq, num_cols):
    r = floor(seq / num_cols)
    c = seq - r * num_cols
    return np.array([[r, c]])

# def create_policy_direction_arrays(model, policy):
#     """
#      define the policy directions
#      0 - up    [0, 1]
#      1 - down  [0, -1]
#      2 - left  [-1, 0]
#      3 - right [1, 0]
#     :param policy:
#     :return:
#     """
#     # action options
#     UP = 0
#     DOWN = 1
#     LEFT = 2
#     RIGHT = 3
#
#     # intitialize direction arrays
#     U = np.zeros((model.num_rows, model.num_cols))
#     V = np.zeros((model.num_rows, model.num_cols))
#
#     for state in range(model.num_states-1):
#         # get index of the state
#         i = tuple(seq_to_col_row(state, model.num_cols)[0])
#         # define the arrow direction
#         if policy[state] == UP:
#             U[i] = 0
#             V[i] = 0.5
#         elif policy[state] == DOWN:
#             U[i] = 0
#             V[i] = -0.5
#         elif policy[state] == LEFT:
#             U[i] = -0.5
#             V[i] = 0
#         elif policy[state] == RIGHT:
#             U[i] = 0.5
#             V[i] = 0
#
#     return U, V


def create_policy_direction_arrays(model, policy, start_state, goal_state):
    """
     define the policy directions
     0 - up    [0, 1]
     1 - down  [0, -1]
     2 - left  [-1, 0]
     3 - right [1, 0]
    :param policy:
    :return:
    """
    # action options
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # intitialize direction arrays
    U = np.empty((model.num_rows, model.num_cols))
    V = np.empty((model.num_rows, model.num_cols))
    U[:] = np.nan
    V[:] = np.nan

    goal_reached = False
    # print('start_state', start_state)
    state = row_col_to_seq(start_state[None, :], model.num_cols) #model.start_state_seq
    # print('goal_state', goal_state)
    goal_state = row_col_to_seq(goal_state[None, :], model.num_cols)
    # print('model.R[state, goal_state]', model.R[state, goal_state])
    # print('model.R[state, goal_state] 2', model.R[:, goal_state])
    it = 0
    while goal_reached is False and it < 100:
        it += 1
        # get index of the state
        i = tuple(seq_to_col_row(state, model.num_cols)[0])
        # define the arrow direction
        # print('policy', policy.shape)
        # print('state', state)
        # print('goal_state', goal_state)
        # print('policy[state, goal_state]', policy[state, goal_state])
        if policy[state, goal_state] == UP:
            U[i] = 0
            V[i] = 0.5
        elif policy[state, goal_state] == DOWN:
            U[i] = 0
            V[i] = -0.5
        elif policy[state, goal_state] == LEFT:
            U[i] = -0.5
            V[i] = 0
        elif policy[state, goal_state] == RIGHT:
            U[i] = 0.5
            V[i] = 0
        state = model._get_state(state, int(policy[state, goal_state][0][0]))
        if model.R[state, goal_state]:
            print('goal_reached', it)
            goal_reached = True
    return U, V



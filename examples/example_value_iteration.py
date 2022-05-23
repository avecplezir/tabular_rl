import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from algorithms.dynamic_programming import value_iteration
from utils.plots import plot_gridworld

###########################################################
#          Run value iteration on a grid world            #
###########################################################

# specify world parameters
# num_cols = 11
# num_rows = 11
# obstructions = np.array([[5,0],[5,1],[5,2],[5,3],[5,4],
#                          [5,6],[5,7],[5,8],[5,9],[5,10]])
# # bad_states = np.array([[1,9],[4,2],[4,4],[7,5],[9,9]])
# # restart_states = np.array([[3,7],[8,2]])
# start_state = np.array([[0,4]])
# # start_state = np.array([[0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6],
# #                         [0,7], [0,8],[0,9], [0,10]])
# # goal_states = start_state + np.array([[10, 0]])
# goal_states = np.array([[9,9]])

num_cols = 3
num_rows = 3
obstructions = np.array([[1,0],[1,2]])
start_state = np.array([[0,0]])
start_states_plot = np.array([[0,0], [0,2]])
# goal_states_plot = np.array([[2,2], [2, 2]])
# start_states_plot = np.array([[0,0]], )
goal_states_plot = np.array([[2,2]], )
goal_states = np.array([[2,2]])

# create model
gw = GridWorld(num_rows=num_rows,
               num_cols=num_cols,
               start_state=start_state,
               goal_states=goal_states)
gw.add_obstructions(obstructed_states=obstructions,
                    # bad_states=bad_states,
                    # restart_states=restart_states
                    )
gw.add_rewards(
               step_reward=0,
               goal_reward=1,
               bad_state_reward=0,
               restart_state_reward=0)
gw.add_transition_probability(p_good_transition=1.,
                              bias=0.)
gw.add_discount(discount=0.9)
model = gw.create_gridworld()

# solve with value iteration
value_function, policy = value_iteration(model, maxiter=100)

# plot the results
path = "../doc/imgs/value_iteration.png"
plot_gridworld(model, value_function=value_function, policy=policy,
               # title="Value iteration",
               path=path,
               start_states=start_states_plot, goal_states=goal_states_plot)

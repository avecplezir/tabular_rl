import torch
import numpy as np

from dataset import ValueDataset
from modules import ValueApproximator


import sys
sys.path.append("..")
import numpy as np
from env.grid_world_multitask import GridWorld
from algorithms.dynamic_programming_multitask import value_iteration
from utils.plots_multitask import plot_gridworld, plot_gridworld_subtasks
import copy

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

###########################################################
#          Run value iteration on a grid world            #
###########################################################

# specify world parameters
num_cols = 4
num_rows = 5
all_states = [ [i, j] for j in range(num_cols) for i in range(num_rows)]
obstructions = [[2,1],[2,2]]
goal_states = np.array([ el for el in all_states if el not in obstructions])
obstructions = np.array(obstructions)
start_states_plot = goal_states[4:5]
goal_states_plot = goal_states[5:]

# create model
gw = GridWorld(num_rows=num_rows,
               num_cols=num_cols,
               start_state=goal_states,
               goal_states=goal_states)
gw.add_obstructions(
                    obstructed_states=obstructions,
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

value_function = value_function[:, :, 0]

device='cuda'

# mean, std = 0., 1.
mean, std = np.mean(goal_states, axis=0), np.std(goal_states, axis=0)

# Prepare dataset
dataset = ValueDataset(value_function,
                       states=goal_states,
                       model=model,
                       mean=mean, std=std
                      )
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# Initialize the MLP
mlp = ValueApproximator(states=goal_states,
         mean=mean, std=std,
         device=device,
         value_function=value_function
        ).to(device)

# Define the loss function and optimizer
# loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3)

# Run the training loop
for epoch in range(0, 1000):  # 5 epochs at maximum
    # Print epoch
    #     print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):
        # Get and prepare inputs
        s, g, s_idx, g_idx = data
        s, g, s_idx, g_idx = s.float().to(device), g.float().to(device), s_idx.long().to(device), g_idx.long().to(
            device)

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        y, v, loss = mlp(s, g, s_idx, g_idx)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()

    if epoch % 10 == 0:
        print('Loss: %.3f' % (current_loss))

    current_loss = 0.0

# Process is complete.
print('Training process has finished.')
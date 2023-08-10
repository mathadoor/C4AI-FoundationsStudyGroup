import numpy as np

# Constants
NUM_ITERATIONS = 1000  # Number of iterations for policy evaluation
GRID_SIZE = (4, 4)  # Size of the grid world

# Initialization of values and rewards for each state in the grid world
values = np.zeros(GRID_SIZE)
rewards = np.ones(GRID_SIZE) * -1  # -1 reward for each state, except the start and terminal state
rewards[-1, -1] = 0  # Terminal state
rewards[0, 0] = 0  # Start state

# Defined actions in terms of changes to row and column indices
actions = {
    0: [1, 1],  # Left
    1: [1, 0],  # Down
    2: [-1, 1], # Right
    3: [-1, 0]  # Up
}


# Iterative Policy Evaluation
for i in range(NUM_ITERATIONS):
    v = values.copy()  # Store current values
    values = np.zeros_like(v)  # Reset new values to zero

    # Loop through possible actions and update state values
    for action, axis in actions.items():
        shifted_values = np.roll(v, axis[0], axis=axis[1])  # Shift values according to the action

        # Ensure that we don't move out of the grid boundaries:
        if action == 0:  # Left
            shifted_values[:, 0] = shifted_values[:, 1]
        if action == 1:  # Down
            shifted_values[0, :] = shifted_values[1, :]
        if action == 2:  # Right
            shifted_values[:, -1] = shifted_values[:, -2]
        if action == 3:  # Up
            shifted_values[-1, :] = shifted_values[-2, :]

        # Start and Terminal state values remain the same
        shifted_values[-1, -1] = 0  # Terminal state
        shifted_values[0, 0] = 0  # Start state

        # Update values based on equal probability policy
        values += 1 / 4 * (rewards + shifted_values)

print(values)  # Display the final values

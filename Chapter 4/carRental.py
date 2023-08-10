import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Constants for the problem
THRESHOLD = 1e-3  # Convergence threshold
MAX_CAR_NUM = 20  # Maximum number of cars in each location
UPPER_BOUND = 9  # Upper bound for Poisson probability
MAX_MOVE = 5  # Maximum number of cars that can be moved overnight
GAMMA = 0.9  # Discount factor
RENTAL_REWARD = 10  # Reward for renting a car
MOVE_REWARD = -2  # Cost for moving a car between locations

# Possible actions, negative means moving cars from location 1 to 2
actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)

# Initialize the value and policy tables
values = np.zeros((MAX_CAR_NUM + 1, MAX_CAR_NUM + 1))
policy = np.zeros_like(values, dtype=np.int32)
poisson_prob = lambda n, l: poisson.pmf(n, l)


def compute_expected_return(state_x, state_y, action):
    """Compute expected return for a given state and action."""
    # Initial reward for moving cars
    ret = abs(action) * MOVE_REWARD

    # Update the number of cars after the action
    num_cars_1 = min(state_x - action, MAX_CAR_NUM)
    num_cars_2 = min(state_y + action, MAX_CAR_NUM)

    # Renting and returning probabilities for location 1
    prob_req_1 = poisson_prob(np.arange(UPPER_BOUND), 3)
    prob_ret_1 = poisson_prob(np.arange(UPPER_BOUND), 3)
    valid_requests_1 = np.clip(np.arange(UPPER_BOUND), 0, num_cars_1)
    returns_1 = np.arange(UPPER_BOUND)
    car_total_1 = np.clip(num_cars_1 + returns_1.reshape(-1, 1) -
                          valid_requests_1.reshape(1, -1), 0, MAX_CAR_NUM)

    # Renting and returning probabilities for location 2
    prob_req_2 = poisson_prob(np.arange(UPPER_BOUND), 4)
    prob_ret_2 = poisson_prob(np.arange(UPPER_BOUND), 2)
    valid_requests_2 = np.clip(np.arange(UPPER_BOUND), 0, num_cars_2)
    returns_2 = np.arange(UPPER_BOUND)
    car_total_2 = np.clip(num_cars_2 + returns_2.reshape(-1, 1) -
                          valid_requests_2.reshape(1, -1), 0, MAX_CAR_NUM)

    # Compute reward from renting cars
    reward = (valid_requests_1.reshape(-1, 1) + valid_requests_2.reshape(1, -1)) * RENTAL_REWARD

    # Obtain the next state's values based on returned and requested cars.
    val = np.take(values, car_total_1, axis=0)
    val = np.take(val, car_total_2, axis=2)
    val = np.swapaxes(val, 1, 2)

    # Probability for each combination of rentals and returns
    prob_ret_req = np.einsum('i,j,k,l->ijkl', prob_ret_1, prob_ret_2, prob_req_1, prob_req_2)

    # Compute the expected return
    ret += np.sum(prob_ret_req * (reward + GAMMA * val))

    return ret


# Policy Iteration method
states = np.array([[i, j] for i in range(MAX_CAR_NUM + 1) for j in range(MAX_CAR_NUM + 1)])

while True:

    # Policy Evaluation step
    old_values = values.copy()
    delta = None
    while delta is None or delta >= THRESHOLD:
        delta = 0
        for i in range(MAX_CAR_NUM + 1):
            for j in range(MAX_CAR_NUM + 1):
                v = values[i, j]
                values[i, j] = compute_expected_return(i, j, policy[i, j])
                delta = max(delta, abs(v - values[i, j]))
        print('Delta: {}'.format(delta))

    # Policy Improvement step
    print('Policy Improvement via Greedy Action Selection')
    for i in range(MAX_CAR_NUM + 1):
        for j in range(MAX_CAR_NUM + 1):
            old_action = policy[i, j]
            action_returns = []
            for action in actions:
                # Check if action is valid
                if (0 <= action <= i) or (-j <= action <= 0):
                    action_returns.append(compute_expected_return(i, j, action))
                else:
                    action_returns.append(-np.inf)
            policy[i, j] = actions[np.argmax(action_returns)]

    # Check for convergence
    if np.mean(np.abs(old_values - values)) < THRESHOLD:
        print('Optimal policy found')
        print(policy)

        # Plotting the value function
        ax = plt.axes(projection='3d')
        x = np.arange(MAX_CAR_NUM + 1)
        y = np.arange(MAX_CAR_NUM + 1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, values, cmap='viridis', edgecolor='none')
        plt.show()
        break

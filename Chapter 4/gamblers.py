import matplotlib.pyplot as plt
import numpy as np

GOAL = 100
STATES = np.arange(GOAL)
HEAD_PROB = 0.5
q_value = np.zeros((GOAL, GOAL))

sweeps_history = []

# value iteration
while True:
    old_state_value = np.max(q_value, axis=1)
    old_q_value = q_value.copy()
    sweeps_history.append(old_state_value)

    for state in STATES[1:GOAL]:
        for action in np.arange(1, min(state, GOAL - state) + 1):
            if state + action >= GOAL:
                opt_next_state_q = 1
            else:
                opt_next_state_q = old_q_value[state + action, :]

            if state - action <= 0:
                pessim_next_state_q = 0
            else:
                pessim_next_state_q = old_q_value[state - action, :]

            q_value[state, action] = np.max(HEAD_PROB * opt_next_state_q + (1 - HEAD_PROB) * pessim_next_state_q)

    delta = abs(np.max(q_value, axis=1) - old_state_value).mean()
    if delta < 1e-12:
        sweeps_history.append(np.max(q_value, axis=1))
        break

# compute the optimal policy
policy = np.argmax(q_value, axis=1)

print(policy)
plt.figure(figsize=(10, 20))

plt.subplot(2, 1, 1)
for sweep, state_value in enumerate(sweeps_history):
    plt.plot(state_value, label='sweep {}'.format(sweep))
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.scatter(STATES, policy)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.savefig('gambler.png')

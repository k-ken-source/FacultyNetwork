from __future__ import print_function, division

import numpy as np
from Graph.teacher import standard_teacher, print_values, print_policy, max_dict, random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ("question", "answer", "blog", "article", "task")



def Deltas():
    
    teach = standard_teacher()

    Q = {}
    states = []
    for i in range(20):
        states.append(i)
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
    print(states)

    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

            # repeat until convergence
    t = 1.0
    deltas = []
    for it in range(1000):
        if it % 10 == 0:
            t += 1e-2
        if it % 100 == 0:
            print("it:", it)

        s = 0
        teach.points = 0

        a, _ = max_dict(Q[s])
        biggest_change = 0
        while not teach.game_over():

            a = random_action(a, eps=1)  # epsilon-greedy
            # random action also works, but slower since you can bump into walls
            # a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            r = teach.act(a)
            s2 = teach.current_state()
            tb = teach.data
            # count = 0
            for col in tb:
                x = tb[col]
                vote = x.values
                upvotes = 0
                for i in range(len(vote)):
                    if vote[i] < 50:
                        for j in range(i + 1, len(vote)):
                            vote[j] += 1
                    else:
                        upvotes += 1

                        for j in range(i + 1, len(vote)):
                            if vote[j] == 0:
                                continue
                            vote[j] -= 1
                # print(count, "votes: ", upvotes)
                # count += 1
                diff_votes = abs(len(vote) - upvotes)
                r = r + diff_votes - upvotes

                # adaptive learning rate
                alpha = ALPHA / update_counts_sa[s][a]
                update_counts_sa[s][a] += 0.005

                # we will update Q(s,a) AS we experience the episode
                old_qsa = Q[s][a]
                # the difference between SARSA and Q-Learning is with Q-Learning
                # we will use this max[a']{ Q(s',a')} in our update
                # even if we do not end up taking this action in the next step
                a2, max_q_s2a2 = max_dict(Q[s2])

                Q[s][a] = Q[s][a] + alpha * (r + GAMMA * max_q_s2a2 - Q[s][a])
                biggest_change = np.abs(Q[s][a])
                # print(biggest_change)
                # biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))
                # we would like to know how often Q(s) has been updated too
                update_counts[s] = update_counts.get(s, 0) + 1

                # next state becomes current state
                s = s2
                a = a2

        deltas.append(biggest_change)

    return deltas
'''
    print("deltas",deltas)
    plt.xlabel("iterations")
    plt.ylabel("Q-value at each iteration")
    plt.plot(deltas)
    plt.show()

# everything below this is policy and other things 

    policy = []
    V = []
    for s in range(20):
        a, max_q = max_dict(Q[s])
        policy.append(a)
        V.append(abs(max_q))

    # what's the proportion of time we spend updating each part of Q?
    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, teach)
    print("\n")

    print("values:")
    print_values(V, teach)
    print("\n")
    print("policy:")
    print_policy(policy, teach)
    print("\n")
'''

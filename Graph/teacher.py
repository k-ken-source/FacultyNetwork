from __future__ import print_function, division

import numpy as np
import pandas as pd
import os


ALL_POSSIBLE_ACTIONS = ("question", "answer", "blog", "article", "task")


class Teacher:

    def __init__(self, states, start):
        # Initialize the faculty list
        path_to_file = './py-dynaq-imp'
        pre = os.path.dirname(os.path.realpath(path_to_file))
        fname = 'reward_data1.csv'
        path = os.path.join(pre, fname)
        reward_data_sheet = pd.read_csv(path)
        faculty_dataframe = pd.DataFrame(reward_data_sheet)
        self.data = faculty_dataframe.loc[:, "post_question_reward":"post_task_reward"]
        self.states = states
        self.points = start

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.state = s

    def set_points(self, points):
        self.points = points

    def is_terminal(self, s):
        return s not in self.actions

    def current_state(self):
        return self.state

    def act(self, action):
        if action in self.actions:
            if action == "question":
                self.points = self.points + 50
            elif action == "answer":
                self.points = self.points + 65
            elif action == "article":
                self.points = self.points + 70
            elif action == "blog":
                self.points = self.points + 70
            elif action == "task":
                self.points = self.points + 100
        if self.points < 1000:
            self.state = int(self.points / 50)
        else:
            self.state = 19

        return self.rewards[self.state][action]

    def game_over(self):
        return self.points > 1000


def print_values(V, t):
    for i in range(20):
        v = V[i]
        if v >= 0:
            print(i, "  %.2f|" % v, end="")
        else:
            print(i, " %.2f|" % v, end="")


def print_policy(Q, t):
    for i in range(20):
        print(i, " ", Q[i], " | ", end="")


def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


def random_action(a, eps=1):
    # we'll use epsilon-soft to ensure all states are visited
    # what happens if you don't do this? i.e. eps=0
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


def standard_teacher():
    t = Teacher(20, 0)
    rewards = {}
    for i in range(0, 20):
        rewards[i] = {"question": 1, "answer": 1, "article": 1, "blog": 1, "task": 1}
    actions = {"question", "answer", "blog", "article", "task"}
    t.set(rewards, actions)
    return t

from collections import deque
import random
import numpy as np

class Memory(object):

    def __init__(self, memory_size, random_seed=123):
        self.memory_size = memory_size
        self.count = 0
        self.buffer = deque()
        self.priorities = deque()
        random.seed(random_seed)

    # add to memory -- without priority
    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.memory_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        return

    # add to memory -- with priority
    def addWithPriority(self, s, a, r, t, s2, p):
        experience = (s, a, r, t, s2)
        if self.count < self.memory_size:
            self.buffer.append(experience)
            self.priorities.append(p)
            self.count += 1
        else:
            self.buffer.popleft()
            self.priorities.popleft()
            self.buffer.append(experience)
            self.priorities.append(p)
        return

    # get size
    def size(self):
        return self.count

    # get sample -- without using priority
    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    # get sample -- using priority
    def sample_batch_priority(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.choices(self.buffer, self.priorities, k=self.count)
        else:
            batch = random.choices(self.buffer, self.priorities, k=batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    # clear memory
    def clear(self):
        self.buffer.clear()
        self.count = 0
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from collections import deque, namedtuple
import random

def make_experience(state: Data, next_state: Data, action: int, reward: float):
    experience = PairData(x_s=state.x, edge_index_s=state.edge_index,
                    x_t=next_state.x,edge_index_t=next_state.edge_index,
                    action=action, reward=reward)
    # Experience = namedtuple('Experience', 'state, next_state, action, reward')
    # e = Experience(state, next_state, action, reward)  
    return experience

class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None, action=None, reward=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.action = action
        self.reward = reward
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class GraphReplayMemory(Dataset):
    def __init__(self, capacity):
        super().__init__()
        self.memory = deque([],maxlen=capacity)

    def len(self) -> int:
        return len(self.memory)

    def append(self, experience: PairData) -> None:
        self.memory.append(experience)

    def get(self, idx: int) -> PairData:
        return self.memory[idx]

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def append(self, state, next_state, action, reward):
        """Save a transition"""
        self.memory.append([state, next_state, action, reward])

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states, next_states, actions, rewards = map(np.asarray, zip(*samples))
        return states, next_states, actions, rewards

    def __len__(self):
        return len(self.memory)

class PERGraphData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None, action=None, reward=None, idx=None, is_weight=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.action = action
        self.reward = reward
        self.idx = idx # the idx in the SumTree 
        self.is_weight = is_weight

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=PairData)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data: PairData):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def __len__(self):
        return self.n_entries

class PrioritizedExperienceReplay():  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        super().__init__()
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def append(self, error, sample):
        
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n): 
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    
    def __len__(self):
        return len(self.tree)
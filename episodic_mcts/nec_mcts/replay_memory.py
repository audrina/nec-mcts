import random
from nec_mcts.transition import Transition

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        '''
        Saves a transition
        :param args: statistics to save to memory
        :return: N/A
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
        Return a training batch.
        :param batch_size: number of training examples to sample
        :return: training batch
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
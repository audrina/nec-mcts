'''
Adapted from: https://github.com/floringogianu/neural-episodic-control/blob/master/data_structures/dnd.py
'''

import torch
import pickle

from sklearn.neighbors import KDTree
from xxhash import xxh64 as xxhs
from nec_mcts.kernel_density_estimate import KernelDensityEstimate
import nec_mcts.config as config

class DND:
    def __init__(self, capacity=config.MEM_CAPACITY, state_repn_nfeat=config.N_FEAT, p_nn=config.PNN):
        '''
        :param capacity: number of states to store in memory
        :param state_repn_nfeat: number of features in state repn produced by CNN (DQN)
        :param p_nn: top p-nearest neighbors to query to perform lookups
        '''
        self.capacity = capacity
        self.state_repn_nfeat = state_repn_nfeat
        self.p_nn = p_nn

        self.memory = {}  # dictionary storing (K_a, V_a) pairs for an action a in A
        self.keys = torch.empty(capacity, state_repn_nfeat).fill_(0)
        self.vals = torch.empty(capacity, 1).fill_(0)
        self.priority = torch.zeros([capacity, 1], dtype=torch.int32)

        self.kde = KernelDensityEstimate()
        self.kd_tree = None
        self.idx = 0
        self.count = 0
        self.old = 0
        self.new = 0
    
    def write(self, h, v, update_rule=None):
        '''
        Writes key-value pair to memory.
        :param h: key produced by the CNN (which received the pixel state as input)
        :param v: value from the DND
        :param update_rule: for updating the values in memory
        :return: N/A
        '''
        h = h.squeeze(0)
        key = self._hash(h)
        is_new_key = key not in self.memory

        if self.idx < self.capacity and is_new_key:
            # new experience, memory not full, append to DND
            self._write(key, h, v, self.idx)
            self.idx += 1
            self.new += 1
        elif not is_new_key:
            # old experience, update its value
            idx_to_update = self.memory[key]
            old_v = self.vals[idx_to_update]
            self.vals[idx_to_update] = update_rule(old_v, v)
            self.old += 1
        else:
            # new experience, full memory, pop least used and append to DND
            write_idx = self._get_least_used()
            t = self.keys[write_idx]
            old_key = self._hash(t)
            self.memory.pop(old_key)
            self._write(key, h, v, write_idx)
            self.new += 1
        self.count += 1

    def lookup(self, h, training=False):
        '''
        Looks up a key in the DND.
        :param h: key produced by the CNN (which received the pixel state as input)
        :param training: boolean specifying whether training mode is on.
        :return: Q-value of the queried key.
        '''
        volatile = not training
        _, knn_indices = self.kd_tree.query(h.data.numpy(), k=self.p_nn)
        mask = torch.from_numpy(knn_indices).long().squeeze()
        h_i = torch.tensor(self.keys[mask])
        v_i = torch.tensor(self.vals[mask])
        if volatile:
            return self._get_q_value(h, h_i, v_i).data
        else:
            return self._get_q_value(h, h_i, v_i)

    def _write(self, key, h, v, idx):
        '''
        Helper writer method.
        :param key: memory key
        :param h: key produced by the CNN (which received the pixel state as input)
        :param v: value from the DND
        :param idx: key-value entry index
        :return:
        '''
        self.memory[key] = idx
        self.keys[idx] = h
        self.vals[idx] = v
        self.priority[idx] = 0

    def _get_q_value(self, h, h_i, v_i):
        '''
        Get the Q-value for a key.
        :param h: queried key
        :param h_i: other keys
        :param v_i: values of the other keys
        :return: value of queried key
        '''
        distances = self.kde.gaussian_kernel(h, h_i)
        weights = self.kde.normalized_kernel(distances)
        return torch.sum(weights * v_i)

    def _get_least_used(self):
        '''
        :return: Index of the least frequently queried memory in the nearest neighbors search.
        '''
        if self.idx < self.capacity:
            return self.priority[:self.idx].min(0)[1][0, 0]
        else:
            return self.priority.min(0)[1][0, 0]

    def _hash(self, h):
        '''
        Hashing function.
        :param h: key
        :return: hash value
        '''
        assert h.ndimension() == 1, "Tensor must be one-dimensional"
        return xxhs(pickle.dumps(h.tolist())).hexdigest()


    def rebuild_tree(self):
        '''
        Re-initializes kd-tree/
        :return: N/A
        '''
        if self.idx < self.capacity:
            self.kd_tree = KDTree(self.keys[:self.idx].numpy())
        else:
            self.kd_tree = KDTree(self.keys.numpy())

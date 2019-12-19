'''
Adapted from: https://github.com/floringogianu/neural-episodic-control/blob/master/agents/nec_agent.py
'''
import torch
import numpy as np
import math
import torch.nn.functional as F

import nec_mcts.config as config
import nec_mcts.utils as utils
from nec_mcts.dnd import DND
from nec_mcts.replay_memory import ReplayMemory
from nec_mcts.input_embedder import Middleware
from nec_mcts.transition import Transition


class NECAgent:
    def __init__(self, state_batch, action_space, dnd):
        self.S = state_batch
        self.A = action_space
        self.action_no = config.N_ACTIONS
        self.name = 'NEC_agent'
        self.dnd = dnd

        # Feature extractor and embedding size
        self.conv = Middleware
        embedding_size = config.EMBEDDING_SIZE

        # DNDs, Memory, N-step buffer
        self.dnds = [DND(dnd.size, embedding_size, dnd.knn_no)
                     for i in range(self.action_no)]
        self.replay_memory = ReplayMemory(capacity=config.MEM_CAPACITY)
        self.N_step = config.HORIZON
        self.N_buff = []

        self.slow_lr = config.SLOW_LR
        self.fast_lr = config.FAST_LR
        self.optimizer = torch.optim.Adam(self.conv.parameters(), lr=config.SLOW_LR)
        self.optimizer.zero_grad()
        self.update_q = utils.update_rule(config.FAST_LR)

        # Temp data, flags, stats, misc
        self._key_tmp = None
        self.knn_ready = False
        self.initial_val = 0.1
        self.max_q = -math.inf

    def evaluate_policy(self, state):
        '''
        Performs a forward operation through the neural net feature
        extractor and uses the resulting representation to compute the k
        nearest neighbors in each of the DNDs associated with each action.

        Returs the action with the highest weighted value between the
        k nearest neighbors.
        :param state: batch of states
        :return: action
        '''
        state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
        h = self.conv(state, config.BOARD_SIZE)
        self._key_tmp = h

        # query each DND for q values and pick the largest one.
        if np.random.uniform() > config.EPS_DECAY:
            v, action = self._query_dnds(h)
            self.max_q = v if self.max_q < v else self.max_q
            return action
        else:
            return self.A.sample()

    def improve_policy(self, _state, _action, reward, state, done):
        '''policy evaluation'''
        self.N_buff.append((_state, self._key_tmp, _action, reward))

        R = 0
        if self.knn_ready and ((len(self.N_buff) == self.N_step) or done):
            if not done:
                # compute Q(t + N)
                state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
                h = self.conv(state, config.BOARD_SIZE)
                R, _ = self._query_dnds(h)

            for i in range(len(self.N_buff) - 1, -1, -1):

                s = self.N_buff[i][0]
                h = self.N_buff[i][1]
                a = self.N_buff[i][2]
                R = self.N_buff[i][3] + 0.99 * R

                # write to DND
                self.dnds[a].write(h.data, R, self.update_q)
                # print("%3d, %3d, %3d  |  %0.3f" % (self.step_cnt, i, a, R))

                # append to experience replay
                self.replay_memory.push(s, a, R)

            self.N_buff.clear()

            for dnd in self.dnds:
                dnd.rebuild_tree()

            # get batch of transitions
            transitions = self.replay_memory.sample(config.BATCH_SIZE)
            batch = self._batch2torch(transitions)
            # compute gradients
            self._accumulate_gradient(*batch)
            # optimize
            self._update_model()

    def _query_dnds(self, h):
        q_vals = torch.zeros(config.N_ACTIONS, 1).fill_(0)
        for i, dnd in enumerate(self.dnds):
            q_vals[i] = dnd.lookup(h)
        return q_vals.max(0)[0][0, 0], q_vals.max(0)[1][0, 0]

    def _accumulate_gradient(self, states, actions, returns):
        '''
        Compute gradient
        :param states:
        :param actions:
        :param returns: QN(s,a)
        :return:
        '''
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        returns = torch.tensor(returns)

        # Compute Q(s, a)
        features = self.conv(states)

        v_variables = []
        for i in range(config.BATCH_SIZE):
            act = actions[i].data[0]
            v = self.dnds[act].lookup(features[i].unsqueeze(0), training=True)
            v_variables.append(v)

        q_values = torch.stack(v_variables)

        loss = F.smooth_l1_loss(q_values, returns)
        loss.data.clamp(-1, 1)

        # Accumulate gradients
        loss.backward()

    def _update_model(self):
        for param in self.conv.parameters():
            param.grad.data.clamp(-1, 1)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def _heat_up_dnd(self, h):
        # fill the dnds with knn_no * (action_no + 1)
        action = np.random.randint(self.action_no)
        self.dnds[action].write(h, self.initial_val, self.update_q)
        return action

    def _batch2torch(self, batch, batch_sz=None):
        '''
        From a batch of transitions (s0, a0, Rt)
        get a batch of the form state=(s0,s1...), action=(a1,a2...),
        Rt=(rt1,rt2...)
        Inefficient. Adds 1.5s~2s for 20,000 steps with 32 agents.
        :param batch: batch of transitions (s0, a0, Rt)
        :param batch_sz: batch size
        :return: list of torch states, actions, rewards.
        '''
        batch_sz = len(batch) if batch_sz is None else batch_sz
        batch = Transition(*zip(*batch))

        states = [torch.from_numpy(s).unsqueeze(0) for s in batch.state]
        state_batch = torch.stack(states).type(torch.float)
        action_batch = torch.tensor(batch.action, dtype=torch.long)
        rt_batch = torch.tensor(batch.Rt, dtype=torch.float)
        return [state_batch, action_batch, rt_batch]

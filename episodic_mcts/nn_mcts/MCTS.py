import math
import numpy as np
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        return probs


    def search(self, canonicalBoard):
        """
        Performs one iteration of the MCTS. 

        Action is chosen based on UCT or PUCT. The state is expand and simulated as 
        leaf node is first visited. The neural network is called to return the inital
        policy and value for the node as rollout step. Then the value is back
        progagated to the upper path. The values of Ns, Nsa, Qsa are updated.

        Returns the negative of the value of the current canonicalBoard.
        """
        # Get the state of this canonical board
        s = self.game.stringRepresentation(canonicalBoard)

        # Check if s is the terminal node
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

        # terminal node
        if self.Es[s]!=0: 
            return -self.Es[s] 

        # leaf node
        if s not in self.Ps: 
            return self.rollout(s, canonicalBoard)

        # inside node
        a = self.PUCT(s) # Can change to PUCT/UCT
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)
        self.Pback_propagation(s, a, v) # Can change to Pback_propagation/back_propagation
        return -v


    def UCT(self, s):
        """
        The UCT of the MCTS.

        Return the action with highest upper confidence bound.
        """
        valids = self.Vs[s]
        best_u = -float('inf')
        best_act = -1

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)]/self.Nsa[(s,a)] + math.sqrt(2) * math.sqrt(
                        math.log(self.Ns[s])/self.Nsa[(s,a)])
                else:
                    u = float('inf')
                
                if u > best_u:
                    best_u = u
                    best_act = a
        return best_act


    def PUCT(self, s):
        """
        The PUCT of MCTS based on alpha zero.

        Return the action with highest polynomial upper confidence bound.
        """
        valids = self.Vs[s]
        best_u = -float('inf')
        best_act = -1

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)
                if u > best_u:
                    best_u = u
                    best_act = a
        return best_act


    def rollout(self, s, canonicalBoard):
        """
        The rollout for MCTS.

        Return the reward for the s.
        """
        self.Ps[s], v = self.nnet.predict(canonicalBoard)
        # Update the policy of the state
        valids = self.game.getValidMoves(canonicalBoard, 1)
        self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
        sum_Ps_s = np.sum(self.Ps[s])

        if sum_Ps_s > 0:
            self.Ps[s] /= sum_Ps_s    # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable
            print("All valid moves were masked, do workaround.")
            self.Ps[s] = self.Ps[s] + valids
            self.Ps[s] /= np.sum(self.Ps[s])

        self.Vs[s] = valids
        self.Ns[s] = 0
        return -v


    def back_propagation(self, s, a, v):
        """
        The back propoagation for MCTS. 

        Updata the new reward to upper path.
        """
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] += v
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1


    def Pback_propagation(self, s, a, v):
        """
        The back propagation of MCTS based on alpha zero.

        Updata the new reward to upper path.
        """
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1

# [END]

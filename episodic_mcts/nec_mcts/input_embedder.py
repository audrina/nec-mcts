'''
Network structure organization: https://nervanasystems.github.io/coach/components/agents/value_optimization/nec.html

State in algo-zero-general/Othello context is the canonical form of the board
g = OthelloGame(n)
attributes/methods:
- g.n --> board size (n x n)
- g.getCanonicalForm(g.getInitBoard(), curPlayer=1) --> return state = player*board (board = np.array(b.pieces))
--> so the observed state is obtained from the OthelloGame class
'''
from dqn import DQN
import config

class Embedder:
    def __init__(self, state_batch, n):
        self.state_batch = state_batch
        self.n = n
        self.cnn = DQN(n, n, config.EMBEDDING_SIZE)

class Middleware(Embedder):
    def __init__(self, state_batch, n):
        super().__init__(state_batch, n)

    def extract_embedding(self):
        '''
        :return: state embedding from DQN as returned by the input embedder.
        '''
        return self.cnn(self.state_batch)

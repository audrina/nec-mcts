from nn_mcts.othello.OthelloGame import OthelloGame

# DQN hyperparams
EMBEDDING_SIZE = 60

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

SLOW_LR = 0.001
FAST_LR = 0.1

# DND memory settings
MEM_CAPACITY = 50000
N_FEAT = 60
PNN = 50
HORIZON = 1000

# Othello game settings
BOARD_SIZE = 8
G = OthelloGame(BOARD_SIZE)
N_ACTIONS = G.getActionSize()


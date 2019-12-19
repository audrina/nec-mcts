from nn_mcts.Coach import Coach
from nn_mcts.othello.OthelloGame import OthelloGame
from nn_mcts.othello.NNet import NNetWrapper as nn
from nn_mcts.utils import *

"""
This script is used to self train the MCTS is several iterations and store the 
best iteration for the game.
"""

args = dotdict({
    'numIters': 20,             # Number of iteration for training.
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/20_8checkpoints/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":
    g = OthelloGame(8) # Define the game
    nnet = nn(g) # Define the NN

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn() # Do the self play and pitting

# [END]

import numpy as np
from os import path
import sys
import os
import re
import matplotlib.pyplot as plt
from nn_mcts.utils import *

"""
This script is used to plot the time and win for each iteration.
Need at least one input file to analyze which is the output file of play.py.
The output of main.py could also be the input file but need pass in with the output file of play.py.
"""
args = dotdict({ 
    'FILE':"./20_8Othello", # File to store plots
    'AVERAGETIME': "/20_8Average_time.png", # plots names
    'FRACTIONWIN': "/20_8Fraction_win.png",
    'TRAINTIME': "/20_8Train_time.png",
})

if __name__ == "__main__":
    # Check if the folder exists
    assert(os.path.isdir(args.FILE))

    # Check if input a file name
    if len(sys.argv) < 2:
        print("Need input a file")
        sys.exit()

    # Check if the file exist
    filename = sys.argv[1]
    assert(os.path.isfile(filename))

    # Get the name of the output files for the plot
    AVERAGETIME = args.FILE + args.AVERAGETIME
    FRACTIONWIN = args.FILE + args.FRACTIONWIN
    TRAINTIME = args.FILE + args.TRAINTIME

    # Get the data to plot from input file
    f = open(filename, 'r')
    data = {}
    iteration = ""
    player = ""
    for line in f.readlines():
        # read the lines
        if "Iteration" in line:
            iteration = line[:-1]
            data[iteration] = {}
        elif "Random" in line:
            player = 'random'
            data[iteration][player] = {}
            data[iteration][player]['time'] = []
        elif "Greedy" in line:
            player = 'greedy'
            data[iteration][player] = {}
            data[iteration][player]['time'] = []
        elif "time" in line:
            time_used = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            data[iteration][player]['time'].append(time_used)
        elif "(" in line:
            data[iteration][player]['win'] = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])/100
            data[iteration][player]['lose'] = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[1])/100
            data[iteration][player]['draw'] = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[2])/100
    total_game = data[iteration][player]['win'] + data[iteration][player]['lose'] + data[iteration][player]['draw']
    f.close()

    # Store the data for plot
    random_t = []
    random_w = []
    greedy_t = []
    greedy_w = []
    for it, v in data.items():
        # Random player
        times = v['random']['time']
        random_t.append(sum(times)/len(times))
        random_w.append(v['random']['win'])
        
        # Greedy player
        times = v['greedy']['time']
        greedy_t.append(sum(times)/len(times))
        greedy_w.append(v['greedy']['win'])

    # Plot for average time and win for each NN vs players
    # Time for each iteration of play in play.py
    plt.plot(random_t, "g", greedy_t, "r")
    plt.xticks(np.arange(0,len(data)+1,5))
    plt.legend(labels = ['random', 'greedy'])
    plt.xlabel('Iteration')
    plt.ylabel('Average time')
    plt.title('Average time used for each iteration\nMCTS vs. random/greedy player')

    plt.savefig(AVERAGETIME, dpi=300)

    plt.show()

    # Number of win for MCTS player with other players
    plt.plot(random_w, "g", greedy_w, "r")
    plt.xticks(np.arange(0,len(data)+1,5))
    plt.yticks(np.arange(0.4, 1.01, 0.1))
    plt.legend(labels = ['random', 'greedy'])
    plt.axhline(y=1, color='k', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Fraction of win')
    plt.title('Fraction of win for each iteration\nMCTS vs. random/greedy player')

    plt.savefig(FRACTIONWIN, dpi=300)

    plt.show()

    # Get the time of training if there are two input files.
    if len(sys.argv) < 3:
        sys.exit()
    filename = sys.argv[2]

    # Check if the file exist
    assert(os.path.isfile(filename))

    # Get the data in file
    f = open(filename, 'r')
    train_time = []
    for line in f.readlines():
        if "Total:" in line:
            time = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])/60
            train_time.append(time)
    f.close()

    # Time used for train
    plt.plot(train_time, "g")
    plt.xticks(np.arange(0, len(train_time),5),np.arange(1, len(train_time)+1,5))
    plt.xlabel('Iteration')
    plt.ylabel('Time(min)')
    plt.title('Train time for each iteration')

    plt.savefig(TRAINTIME, dpi=300)

    plt.show()

    # For further analysis, store all the result to one file
    with open(args.FILE+'/results.txt', 'w') as filehandle:
        filehandle.write('%s\n' % args.FILE)
        filehandle.write('Fraction of win vs. random\n')
        filehandle.writelines("%s\n" % win for win in random_w)
        # filehandle.write('Fraction of win vs. greedy\n')
        # filehandle.writelines("%s\n" % win for win in greedy_w)
        filehandle.write('Time use for play vs. random\n')
        filehandle.writelines("%s\n" % time for time in random_t)
        # filehandle.write('Time use for play vs. greedy\n')
        # filehandle.writelines("%s\n" % time for time in greedy_t)
        filehandle.write('Time use for training\n')
        filehandle.writelines("%s\n" % t for t in train_time)
        filehandle.close()

# [END]
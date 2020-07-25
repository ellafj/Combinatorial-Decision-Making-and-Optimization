from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#np.set_printoptions(linewidth=50)

from util import *


def SMT(w, h, nPres, dims):
    s = Solver()                    # Initializing solver

    # Making grid as in N-queens example for each potential placing of presents
    # Grid[0] = all placements of present 0
    grid = [[[f'{i}{j}{k}' for i in range(w)]for j in range(h)]for k in range(nPres)]

    for present in range(nPres):
        dim = dims[present]

        # Initializing constraint for where presents can be placed on paper
        remainingW = w - dim[0]     # Remaining width
        remainingH = h - dim[1]     # Remaining height
        temp = []




if __name__ == '__main__':
    w, h, nPres, dims = readFile('./Instances/12x12.txt')


from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#np.set_printoptions(linewidth=50)


def readFile(filename):
    f = open(filename, 'r')
    lines = f.readlines()

    dims = []
    w, h, nPres, iter = 0, 0, 0, 0

    for line in lines:
        line = line.split()

        if iter == 0:
            w = int(line[0])
            h = int(line[1])

        elif iter == 1:
            nPres = int(line[0])

        else:
            if len(line) != 0:
                dims.append([int(line[0]), int(line[1])])

        iter += 1

    f.close()
    return w, h, nPres, dims

def writeSolutions(filename, w, h, nPres, dims, leftCorners, time):
    newname = filename.replace('.txt', 'sol.txt')
    f = open('./Solutions/' + newname, 'w')
    f.write('%d %d\n' % (w, h))
    f.write('%d\n' % nPres)
    for i in range(nPres):
        corner = leftCorners[i]
        dim = dims[0]
        f.write('%d %d %s %s\n' % (dim[0], dim[1], corner[0], corner[1]))
    f.write('Solved in %d seconds' % time)
    f.close()

def printPaper(w, h, dist):
    sns.heatmap(dist, linewidth=0.5, annot=True, cbar=False)
    plt.title('Placement of presents')
    plt.ylabel('Height of paper')
    plt.ylim(0,h)
    plt.xlabel('Width of paper')
    plt.xlim(0,w)
    plt.savefig('%dx%d.pdf' % (w, h))

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

def writeSolutions(filename, w, h, nPres, dims, leftCorners, time, dir, type):
    newname = filename.replace('.txt', 'sol.txt')
    f = open(dir + newname, 'w')
    f.write('%d %d\n' % (w, h))
    f.write('%d\n' % nPres)
    if type == 'SAT/SMT':
        for i in range(nPres):
            corner = leftCorners[i].split('-')
            dim = dims[i]
            f.write('%d %d %s %s\n' % (dim[0], dim[1], corner[0], corner[1]))
    elif type == 'CP':
        """dims = np.array(dims)
        dims = dims.flatten()
        invalidTypes = ['[', ' ', ',', ']']
        corners = [int(i) for i in leftCorners if i not in invalidTypes]
        print('leftCorners', corners)
        print('dims', dims)
        print('len', len(corners))
        for i in range(0,len(corners),2):
            print('i', i)
            f.write('%d %d %s %s\n' % (dims[i], dims[i+1], corners[i], corners[i+1]))
        """
        for i in range(nPres):
            corner = leftCorners[i]
            dim = dims[i]
            f.write('%d %d %s %s\n' % (dim[0], dim[1], corner[0], corner[1]))
    f.write('Solved in %d seconds' % time)
    f.close()

def printPaper(w, h, dist, dir):
    sns.heatmap(dist, linewidth=0.5, cbar=False, annot=True, annot_kws={"size": 5})
    plt.title('Placement of presents')
    plt.ylabel('Height of paper')
    plt.ylim(0,h)
    plt.xlabel('Width of paper')
    plt.xlim(0,w)
    plt.savefig(dir + '/%dx%d.pdf' % (w, h))
    plt.close()

def make_dzn():
    directory = './Instances/'
    dir = './dzn_Instances/'
    for filename in os.listdir(directory):
        if filename != '.DS_Store':
            w, h, nPres, dims = readFile(directory + filename)
            newname = filename.replace('txt', 'dzn')
            f = open(dir + newname, 'w')
            f.write('w=%d;\n' % w)
            f.write('h=%d;\n' % h)
            f.write('nPres=%d;\n' % nPres)
            f.write('dims=\n')
            for i in range(nPres):
                dim = dims[i]
                if i == 0:
                    f.write('[|%d, %d\n' % (dim[0], dim[1]))
                elif i == nPres-1:
                    f.write('|%d, %d|];' %(dim[0], dim[1]))
                else:
                    f.write('|%d, %d\n' %(dim[0], dim[1]))
            f.close()


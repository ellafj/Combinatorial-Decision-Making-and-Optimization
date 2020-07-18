from z3 import *
import numpy as np

#x = Int('x')
#y = Int('y')
#print(simplify(x + y + 2*x + 3))
#print(simplify(x < y + x + 2))
#print(simplify(And(x + 1 >= 3, x**2 + x**2 + y**2 + 2 >= 5)))

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

print(readFile('./Instances/8x8.txt'))

def initGrid(h, w, nPres):
    x = np.arange(w)
    y = np.arange(h)
    pres = np.arange(nPres)
    grid = np.zeros((h,w,nPres))
    for i in range(len(x)):
        for j in range(len(y)):
            for it in range(len(pres)):
                grid[i,j,it] = str(i)+str(j)+str(it)

    print('grid', grid)


    #grid[0] = [i for i in range(w)]
    #print(grid[0])
    #for i in range(nPres):


    potGrid = [] # Potential grid
    x = [i for i in range(w)]
    y = [i for i in range(h)]
    pres = [i for i in range(nPres)]

    potGrid.append(x)
    potGrid.append(y)
    potGrid.append(pres)




initGrid(3,3,2)

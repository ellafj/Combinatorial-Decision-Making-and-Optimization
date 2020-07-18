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

# Making grid as in N-queens example
def initGrid(h, w, nPres):
    grid = np.zeros((h,w,nPres))
    for i in range(w):
        for j in range(h):
            for pres in range(nPres):
                grid[i,j,pres] = Bool(str(i)+str(j)+str(pres))

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

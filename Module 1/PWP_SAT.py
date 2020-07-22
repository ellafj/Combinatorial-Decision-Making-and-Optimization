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

def SAT(h, w, nPres, dims):
    s = Solver()                    # Initializing solver

    # Making grid as in N-queens example for each potential placing of presents
    # Grid[0] = all placements of present 0
    grid = [[[Bool(f'{i}{j}{k}') for i in range(w)]for j in range(h)]for k in range(nPres)]
    intGrid = [[[f'{i}{j}{k}' for i in range(w)]for j in range(h)]for k in range(nPres)]

    overlapConst = []

    for present in range(nPres):
        dim = dims[present]
        print('\n current present:', present+1, dim)

        # Initializing constraint for where presents can be placed on paper
        placementConst = []         # Constraint for the potential placements of the presents on the grid
        remainingW = w - dim[0]     # Remaining width
        remainingH = h - dim[1]     # Remaining height

        for x in range(remainingW+1):
            for y in range(remainingH+1):
                place = []          # Resetting placements as we begin with new present
                for i in range(dim[0]):
                    for j in range(dim[1]):

                        # And-command as a present will use all the gridpoints, not just some
                        place.append(grid[present][y+j][x+i])

                # Or-command as we only need *one* placement of present
                placementConst.append(And(*place))
        #print(placementConst)

        s.add(Or(*placementConst))           # Adding constraint to the solver
        print('where', Or(*placementConst), '\n')
        """
        # Constraint to make sure present is only placed once
        onceConst = []
        n = len(placementConst)
        for i in range(n):
            for j in range(i):
                onceConst.append(Not(*[And(placementConst[i], placementConst[j])]))

            
            if n != 0:
                print('n', n)
                onceConst.append(And(placementConst[i], placementConst[i+n-1]))
                n -= 1
        if len(onceConst) == 1:
            s.add(onceConst)
            print('once1', onceConst, '\n')
        else:
            s.add(And(*onceConst))
            print('once2', (And(*onceConst)), '\n')"""


        # Initializing constraint that ensures that presents do not overlap
        for x in range(w):
            for y in range(h):
                for nextPres in range(present):
                    #oconst = Not(Or(*[And(grid[present][x][y], grid[nextPres][x][y])]))
                    #s.add(oconst)
                    #print(oconst)
                    overlapConst.append(Not(*[And([grid[present][y][x], grid[nextPres][y][x]])]))
                    #overlapConst.append(Not(Or(*[And([grid[present][y][x], grid[nextPres][y][x]])])))
                    #s.add(Not(Or(*[And([grid[present][y][x], grid[nextPres][y][x]])])))
                    #print('0', Not(Or(*[And([grid[present][y][x], grid[nextPres][y][x]])])))
        #if present != 0:
            #s.add(overlapConst)
            #s.add(And(*(overlapConst)))
            #print('overlap',And(*(overlapConst)), '\n')

    s.add(And(*(overlapConst)))
    print('overlap',And(*(overlapConst)), '\n')

    print('hello')
    #print(s)
    print(s.check())
    print(s.unsat_core())
    m = s.model()

    return m, grid, intGrid

def collectSolution(m, grid, intGrid):
    sol = []
    area = []
    dist = []

    for y in range(h):
        dis = []
        for x in range(w):
            for pres in range(nPres):
                if m[grid[pres][y][x]]:
                    #print(pres, end=' ')
                    dis.append(pres)
        dist.append(dis)
        #print('\n')

    #dist = dist[::-1]
    print('h',dist)

    for pres in range(nPres):
        for x in range(w):
            for y in range(h):
                if m[grid[pres][y][x]]:
                    area.append(intGrid[pres][y][x])
        sol.append(area)
        area = []

    print(sol)
    leftCorners = []

    for pres in sol:
        leftCorners.append(min(pres))

    print(dims)
    print('left',leftCorners)

    return dist



def printPaper(w, h, dist):
    sns.heatmap(dist, linewidth=0.5, annot=True, cbar=False)
    plt.title('Placement of presents')
    plt.ylabel('Height of paper')
    plt.ylim(0,h)
    plt.xlabel('Width of paper')
    plt.xlim(0,w)
    plt.show()



if __name__ == '__main__':
    w, h, nPres, dims = readFile('./Instances/11x11.txt')
    print(w,h,nPres,dims)
    m, grid, intGrid = SAT(w, h, nPres, dims)
    dist = collectSolution(m, grid, intGrid)
    printPaper(w,h,dist)


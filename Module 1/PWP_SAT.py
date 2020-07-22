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

    placementConst = []         # Constraint for the potential placements of the presents on the grid
    overlapConst = []
    mostOnce = []
    leastOnce = []

    for present in range(nPres):
        dim = dims[present]
        print('\n current present:', present+1, dim)

        # Initializing constraint for where presents can be placed on paper
        remainingW = w - dim[0]     # Remaining width
        remainingH = h - dim[1]     # Remaining height

        temp = []

        # Constraint for connecting gridpoints together for shapes of presents
        for x in range(remainingW+1):
            for y in range(remainingH+1):
                place = []          # Resetting placements as we begin with new present
                for i in range(dim[0]):
                    for j in range(dim[1]):

                        # And-command as a present will use all the gridpoints, not just some
                        place.append(grid[present][y+j][x+i])

                # Or-command as we only need *one* placement of present REWRITE THIS COMMENT, NOW WRONG
                temp.append(And(*place))

        placementConst.append(Or(*temp))

        for i in range(len(temp)):
            for j in range(i):
                mostOnce.append(Not(And(*[temp[i], temp[j]])))

        # Initializing constraint that ensures that presents do not overlap
        for x in range(w):
            for y in range(h):
                for nextPres in range(present):
                    overlapConst.append(Not(*[And([grid[present][y][x], grid[nextPres][y][x]])]))


    # Constraint to make sure present is at least placed once
    n = len(placementConst)
    for i in range(n):
        for j in range(i):
            leastOnce.append(*[And(placementConst[i], placementConst[j])])

    # Adding constraints to the solver
    s.add(And(*placementConst))
    s.add(And(*overlapConst))
    s.add(And(*mostOnce))
    s.add(And(*leastOnce))

    print('where', And(*placementConst), '\n')
    print('overlap',And(*overlapConst), '\n')
    print('most once', And(*mostOnce), '\n')
    print('least once', And(*leastOnce), '\n')

    print('hello')
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


from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(linewidth=50)

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

#print(readFile('./Instances/8x8.txt'))


def SAT(h, w, nPres, dims):
    s = Solver()                    # Initializing solver

    # Making grid as in N-queens example for each potential placing of presents
    # Grid[0] = all placements of present 0
    grid = [[[Bool(f'{i}{j}{k}') for i in range(w)]for j in range(h)]for k in range(nPres)]
    intGrid = [[[f'{i}{j}{k}' for i in range(w)]for j in range(h)]for k in range(nPres)]

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
                        place.append(grid[present][x+i][y+j])

                # Or-command as we only need *one* placement of present
                placementConst.append(And(place))
        #print(placementConst)

        s.add(Or(placementConst))           # Adding constraint to the solver

        # Initializing constraint that ensures that presents do not overlap
        overlapConst = []
        for x in range(w):
            for y in range(h):
                for nextPres in range(present):
                    #oconst = Not(Or(*[And(grid[present][x][y], grid[nextPres][x][y])]))
                    #s.add(oconst)
                    #print(oconst)
                    overlapConst.append(And(*[grid[present][x][y], grid[nextPres][x][y]]))
                s.add(Not(Or(overlapConst)))
                #print('0',Not(Or(overlapConst)))
    print('hello')
    print(s)
    print(s.check())
    print(s.unsat_core())
    m = s.model()
    sol = []
    area = []
    dist = []

    for x in range(w):
        dis = []
        for y in range(h):
            for pres in range(nPres):
                if m[grid[pres][x][y]]:
                    print(pres, end=' ')
                    dis.append(pres)
        dist.append(dis)
        print('\n')

    dist = dist[::-1]
    print(dist)

    for pres in range(nPres):
        for x in range(w):
            for y in range(h):
                if m[grid[pres][x][y]]:
                    area.append(intGrid[pres][x][y])
        sol.append(area)
        area = []

    print(sol)
    leftCorners = []

    for pres in sol:
        leftCorners.append(min(pres))

    print(dims)
    print('left',leftCorners)

    #plt.imshow(dist, cmap='BuPu')#, levels=levels)
    #plt.colorbar()
    #plt.xlim(0,w)
    #plt.ylim(0,h)
    ax = sns.heatmap(dist, linewidth=0.5, annot=True, cbar=False)#, xticklabels=2, yticklabels=2)
    #ax.invert_yaxis()
    plt.title('Placement of presents on paper')
    plt.ylabel('Height of paper')
    plt.ylim(0,h)
    plt.xlabel('Width of paper')
    plt.xlim(0,w)
    plt.show()
    #plt.contourf(dist, cmap='hot')#, levels=levels)
    #plt.show()


if __name__ == '__main__':
    w, h, nPres, dims = readFile('./Instances/8x8.txt')
    print(w,h,nPres,dims)
    SAT(w, h, nPres, dims)


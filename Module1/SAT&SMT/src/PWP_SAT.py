from z3 import *
from util import *
import time


def SAT(w, h, nPres, dims):
    s = Solver()                    # Initializing solver
    M = {}                          # Initializing map of constraints

    # Making grid as in N-queens example for each potential placing of presents
    # Grid[0] = all placements of present 0
    grid = [[[Bool(f'{x}-{y}-{p}') for x in range(w)]for y in range(h)]for p in range(nPres)]
    intGrid = [[[f'{x}-{y}-{p}' for x in range(w)]for y in range(h)]for p in range(nPres)]

    placementConst = []         # Constraint for the potential placements of the presents on the grid
    overlapConst = []
    mostOnce = []
    leastOnce = []

    for present in range(nPres):
        print('\n')
        print('current present:', present)
        dim = dims[present]

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

        # Constraint that ensures present is placed at most once
        for i in range(len(temp)):
            for j in range(i):
                mostOnce.append(Not(And(*[temp[i], temp[j]])))


        # Constraint that ensures that presents do not overlap
        for x in range(w):
            for y in range(h):
                for nextPres in range(present):
                    overlapConst.append(Not(*[And([grid[present][y][x], grid[nextPres][y][x]])]))

    # Constraint that ensures present is at least placed once
    n = len(placementConst)
    for i in range(n):
        for j in range(i):
            leastOnce.append(*[And(placementConst[i], placementConst[j])])

    s.assert_and_track(And(*placementConst), 'place')
    s.assert_and_track(And(*overlapConst), 'overlap')
    s.assert_and_track(And(*mostOnce), 'mostOnce')
    s.assert_and_track(And(*leastOnce), 'leastOnce')

    print('Solving \n')
    print(s.check())
    print(s.unsat_core())
    m = s.model()

    return m, grid, intGrid

def SAT_rotation(w, h, nPres, dims):
    s = Solver()                    # Initializing solver
    M = {}                          # Initializing map of constraints

    # Making grid as in N-queens example for each potential placing of presents
    # Grid[0] = all placements of present 0
    grid = [[[Bool(f'{x}-{y}-{p}') for x in range(w)]for y in range(h)]for p in range(nPres)]
    intGrid = [[[f'{x}-{y}-{p}' for x in range(w)]for y in range(h)]for p in range(nPres)]

    placementConst = []         # Constraint for the potential placements of the presents on the grid
    overlapConst = []
    mostOnce = []
    leastOnce = []

    for present in range(nPres):
        print('\n')
        print('current present:', present)
        dim = dims[present]

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

                        place.append(grid[present][y+j][x+i]) # Not rotated

                # Or-command as we only need *one* placement of present REWRITE THIS COMMENT, NOW WRONG
                temp.append(And(*place))

        # Constraint for connecting gridpoints together for shapes of presents for rotated presents
        if dim[0] != dim[1]:
            for x in range(w - dim[1]+1):
                for y in range(h - dim[0]+1):
                    place = []          # Resetting placements as we begin with new present

                    for i in range(dim[1]):
                        for j in range(dim[0]):

                            place.append(grid[present][y+j][x+i]) # rotated

                    # Or-command as we only need *one* placement of present REWRITE THIS COMMENT, NOW WRONG
                    temp.append(And(*place))

        placementConst.append(Or(*temp))

        # Constraint that ensures present is placed at most once
        for i in range(len(temp)):
            for j in range(i):
                mostOnce.append(Not(And(*[temp[i], temp[j]])))


        # Constraint that ensures that presents do not overlap
        for x in range(w):
            for y in range(h):
                for nextPres in range(present):
                    overlapConst.append(Not(*[And([grid[present][y][x], grid[nextPres][y][x]])]))

    # Constraint that ensures present is at least placed once
    n = len(placementConst)
    for i in range(n):
        for j in range(i):
            leastOnce.append(*[And(placementConst[i], placementConst[j])])

    s.assert_and_track(And(*placementConst), 'place')
    s.assert_and_track(And(*overlapConst), 'overlap')
    s.assert_and_track(And(*mostOnce), 'mostOnce')
    s.assert_and_track(And(*leastOnce), 'leastOnce')

    print('where', And(*placementConst), '\n')
    print('most once', And(*mostOnce), '\n')

    print('Solving \n')
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
                    dis.append(pres)
        dist.append(dis)

    for pres in range(nPres):
        for x in range(w):
            for y in range(h):
                if m[grid[pres][y][x]]:
                    area.append(intGrid[pres][y][x])
        sol.append(area)
        area = []

    leftCorners = []

    for pres in sol:
        leftCorners.append(pres[0])

    return dist, leftCorners


if __name__ == '__main__':
    directory = './Instances/'
    dir = './SAT_Solutions/'
    for filename in os.listdir(directory):
        if filename != '.DS_Store':
            start = time.time()
            print('Currently working on file:', filename)
            w, h, nPres, dims = readFile(directory + filename)
            m, grid, intGrid = SAT(w, h, nPres, dims)
            dist, leftCorners = collectSolution(m, grid, intGrid)
            now = time.time() - start
            print('time',start, now)
            writeSolutions(filename, w, h, nPres, dims, leftCorners, now, dir, 'SAT')
            printPaper(w,h,dist,dir)


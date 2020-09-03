from z3 import *
import numpy as np
import time
#np.set_printoptions(linewidth=50)

from util import *

def SMT(w, h, nPres, dims):
    s = Solver()                    # Initializing solver
    M = {}                          # Initializing map of constraints


    corners = []
    dist = [[Int(f'A{x}{y}') for x in range(w)] for y in range(h)]

    overlapConst = []

    # Initializing corner values
    for present in range(nPres):
        corners.append([Int('x' + str(present)), Int('y'+ str(present) )])

    for present in range(nPres):
        for coord in range(2):
            # Placement constraint
            s.add(0 <= corners[present][coord])
            if coord == 0:
                s.add(corners[present][coord] + dims[present][coord] <= w)
            else:
                s.add(corners[present][coord] + dims[present][coord] <= h)

         # No overlap constraint
        temp1 = []
        for pastPres in range(present):
            temp2 = []
            for coord in range(2):
                temp2.append(Or(corners[present][coord] + dims[present][coord] <= corners[pastPres][coord],
                                       corners[pastPres][coord] + dims[pastPres][coord] <= corners[present][coord]))
            temp1.append(Or(*temp2))
        if present != 0:
            overlapConst.append(And(*temp1))


    s.assert_and_track(And(*overlapConst), 'overlap')
    print(overlapConst)
    print(s.check())
    print(s.unsat_core())
    m = s.model()

    # Writing solution in printable format
    leftCorners = []

    for present in range(nPres):
        xvalue = m[corners[present][0]].as_long()
        yvalue = m[corners[present][1]].as_long()
        leftCorners.append([xvalue, yvalue])
        dim = dims[present]
        for x in range(dim[0]):
            for y in range(dim[1]):
                dist[yvalue + y][xvalue + x] = present

    return leftCorners, dist

def SMT_rotation(w, h, nPres, dims):
    s = Solver()                    # Initializing solver
    M = {}                          # Initializing map of constraints

    corners = []
    dist = [[Int(f'A{x}{y}') for x in range(w)] for y in range(h)]

    overlapConst = []

    # Initializing corner values
    for present in range(nPres):
        corners.append([Int('x' + str(present)), Int('y'+ str(present) )])

    for present in range(nPres):
        placementN = [] # normal
        placementR = [] # rotated
        for coord in range(2):
            # Placement constraint
            s.add(0 <= corners[present][coord])
            if coord == 0:
                placementN.append(corners[present][coord] + dims[present][coord] <= w)
                placementR.append(corners[present][coord] + dims[present][coord+1] <= w)
            else:
                placementN.append(corners[present][coord] + dims[present][coord] <= h)
                placementR.append(corners[present][coord] + dims[present][coord-1] <= h)

        s.add(Or(And(*placementN),And(*placementR)))
        print(Or(And(*placementN), And(*placementR)))

         # No overlap constraint
        temp1 = []
        for pastPres in range(present):
            temp2 = []
            for coord in range(2):
                temp2.append(Or(corners[present][coord] + dims[present][coord] <= corners[pastPres][coord],
                                       corners[pastPres][coord] + dims[pastPres][coord] <= corners[present][coord]))
            temp1.append(Or(*temp2))
        if present != 0:
            overlapConst.append(And(*temp1))

    s.assert_and_track(And(*overlapConst), 'overlap')
    print(overlapConst)
    print(s.check())
    print(s.unsat_core())
    m = s.model()

    # Writing solution in printable format
    leftCorners = []

    for present in range(nPres):
        xvalue = m[corners[present][0]].as_long()
        yvalue = m[corners[present][1]].as_long()
        leftCorners.append([xvalue, yvalue])
        dim = dims[present]

    return leftCorners, dist


if __name__ == '__main__':
    directory = './Instances/'
    dir = './SMT_Solutions/'
    for filename in os.listdir(directory):
        if filename != '.DS_Store':
            start = time.time()
            print('Currently working on file:', filename)
            w, h, nPres, dims = readFile(directory + filename)
            leftCorners, dist = SMT(w, h, nPres, dims) # Not allowing rotation
            #leftCorners, dist = SMT_rotation(w, h, nPres, dims) # Allowing rotation
            now = time.time() - start
            writeSolutions(filename, w, h, nPres, dims, leftCorners, now, dir, 'SMT')
            printPaper(w,h,dist,dir)



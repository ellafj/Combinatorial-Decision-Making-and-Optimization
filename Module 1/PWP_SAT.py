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


def SAT(h, w, nPres, dims):
    # Making grid as in N-queens example for each potential placing of presents
    # Grid[0] = all placements of present 0
    grid = [[[Bool(f'{i}{j}{k}') for i in range(w)]for j in range(h)]for k in range(nPres)]
    print('grid', grid)

    # Constraints:
    print(dims)
    for present in range(len(dims)):
        dim = dims[present]
        print('current present:', present, dim)

        placements = []             # All the potential placements of the presents on the grid
        remainingH = h - dim[0]     # Remaining height
        remainingW = w - dim[1]     # Remaining width
        startX = 0
        startY = 0

        while remainingH != -1 or remainingW != -1:
            print('remaining:', remainingH, remainingW)
            place = []

            if remainingH != -1 and remainingW != -1:
                for x in range(startX,dim[0]+startX):
                    for y in range(startY,dim[1]+startY):
                        place.append(grid[present][x][y])

                placements.append(place)

                remainingH -= 1
                remainingW -= 1
                startX += 1
                startY += 1

            elif remainingH == -1 and remainingW != -1:
                for x in range(startX,dim[0]+startX):
                    for y in range(startY,dim[1]+startY):
                        place.append(grid[present][x][y])

                placements.append(place)

                remainingW -= 1
                startY += 1

            else:
                for x in range(startX,dim[0]+startX):
                    for y in range(startY,dim[1]+startY):
                        place.append(grid[present][x][y])
                print(place)
                placements.append(place)

                remainingH -= 1
                startX += 1

            print(placements)


if __name__ == '__main__':
    w, h, nPres, dims = readFile('./Instances/8x8.txt')
    SAT(w, h, nPres, dims)

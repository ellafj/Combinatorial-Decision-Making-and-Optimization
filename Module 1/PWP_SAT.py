from z3 import *
import numpy as np
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
        print('\n current present:', present, dim)

        placements = []             # All the potential placements of the presents on the grid
        test = []
        remainingW = w - dim[0]     # Remaining width
        remainingH = h - dim[1]     # Remaining height
        startX = 0
        startY = 0

        for x in range(remainingW+1):
            for y in range(remainingH+1):
                place = []
                te = []
                it = 0
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        place.append(grid[present][x+i][y+j])
                        te.append(x)
                        it += 1
                print(it, '\n')
                print(place)
                placements.append(place)
                test.append(te)


        print(placements)
        print(test)
        print(len(placements[0]))
        print(len(placements[1]))
        print(len(placements[2]))
        print(len(placements[3]))

        """
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
        """

if __name__ == '__main__':
    w, h, nPres, dims = readFile('./Instances/8x8.txt')
    SAT(w, h, nPres, dims)

"""
[[003, 103, 203, 303, 403, 013, 113, 213, 313, 413, 023, 123, 223, 323, 423, 033, 133, 233, 333, 433, 043, 143, 243, 343, 443], 
[103, 203, 303, 403, 503, 113, 213, 313, 413, 513, 123, 223, 323, 423, 523, 133, 233, 333, 433, 533, 143, 243, 343, 443, 543], 
[203, 303, 403, 503, 603, 213, 313, 413, 513, 613, 223, 323, 423, 523, 623, 233, 333, 433, 533, 633, 243, 343, 443, 543, 643], 
[303, 403, 503, 603, 703, 313, 413, 513, 613, 713, 323, 423, 523, 623, 723, 333, 433, 533, 633, 733, 343, 443, 543, 643, 743], 
[013, 113, 213, 313, 413, 023, 123, 223, 323, 423, 033, 133, 233, 333, 433, 043, 143, 243, 343, 443, 053, 153, 253, 353, 453], 
[113, 213, 313, 413, 513, 123, 223, 323, 423, 523, 133, 233, 333, 433, 533, 143, 243, 343, 443, 543, 153, 253, 353, 453, 553], 
[213, 313, 413, 513, 613, 223, 323, 423, 523, 623, 233, 333, 433, 533, 633, 243, 343, 443, 543, 643, 253, 353, 453, 553, 653], 
[313, 413, 513, 613, 713, 323, 423, 523, 623, 723, 333, 433, 533, 633, 733, 343, 443, 543, 643, 743, 353, 453, 553, 653, 753], 
[023, 123, 223, 323, 423, 033, 133, 233, 333, 433, 043, 143, 243, 343, 443, 053, 153, 253, 353, 453, 063, 163, 263, 363, 463], 
[123, 223, 323, 423, 523, 133, 233, 333, 433, 533, 143, 243, 343, 443, 543, 153, 253, 353, 453, 553, 163, 263, 363, 463, 563], 
[223, 323, 423, 523, 623, 233, 333, 433, 533, 633, 243, 343, 443, 543, 643, 253, 353, 453, 553, 653, 263, 363, 463, 563, 663], 
[323, 423, 523, 623, 723, 333, 433, 533, 633, 733, 343, 443, 543, 643, 743, 353, 453, 553, 653, 753, 363, 463, 563, 663, 763], 
[033, 133, 233, 333, 433, 043, 143, 243, 343, 443, 053, 153, 253, 353, 453, 063, 163, 263, 363, 463, 073, 173, 273, 373, 473], 
[133, 233, 333, 433, 533, 143, 243, 343, 443, 543, 153, 253, 353, 453, 553, 163, 263, 363, 463, 563, 173, 273, 373, 473, 573], 
[233, 333, 433, 533, 633, 243, 343, 443, 543, 643, 253, 353, 453, 553, 653, 263, 363, 463, 563, 663, 273, 373, 473, 573, 673], 
[333, 433, 533, 633, 733, 343, 443, 543, 643, 743, 353, 453, 553, 653, 753, 363, 463, 563, 663, 763, 373, 473, 573, 673, 773]]

"""

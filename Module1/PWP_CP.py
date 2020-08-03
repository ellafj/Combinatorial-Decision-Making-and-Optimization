import pymzn
import time
import numpy as np
from util import *



place = pymzn.minizinc('PWP_CP_global.mzn', 'fake_Instances/8x8copy.dzn', output_mode='item')

print('place', place)

def make_dist(w,h,corners):
    invalidTypes = ['[', ' ', ',', ']']
    intCorners = [int(i) for i in corners if i not in invalidTypes]
    corners = np.reshape(intCorners, (-1, 2))
    dist = np.zeros((w,h))
    for i in range(nPres):
        dim = dims[i]
        start = corners[i]
        print(start)
        for x in range(dim[0]):
            for y in range(dim[1]):
                dist[start[0]+x-1][start[1]+y-1] = i
    print(dist)
    return dist

if __name__ == '__main__':
    directory = './dzn_Instances/'
    txtdir = './Instances/'
    dir = './CP_Solutions/'
    for filename in os.listdir(directory):
        if filename != '.DS_Store':
        #if filename == '8x8.dzn':
            txtname = filename.replace('dzn', 'txt')
            start = time.time()
            print('Currently working on file:', filename)
            w, h, nPres, dims = readFile(txtdir + txtname)
            corners = pymzn.minizinc('PWP_CP_global.mzn', directory + filename, output_mode='item')
            now = time.time() - start
            print('time',start, now)
            writeSolutions(txtname, w, h, nPres, dims, corners[0], now, dir, 'CP')
            dist = make_dist(w,h,corners[0])
            printPaper(w,h,dist,dir)

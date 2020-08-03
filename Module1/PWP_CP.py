import pymzn
import time
import numpy as np
from util import *


def make_dist(w,h,corners):
    #invalidTypes = ['[', ' ', ',', ']']
    #intCorners = [int(i) for i in corners if i not in invalidTypes]
    #corners = np.reshape(intCorners, (-1, 2))
    dist = np.zeros((w,h))
    for i in range(nPres):
        dim = dims[i]
        start = corners[i]
        print(start)
        for x in range(dim[0]):
            for y in range(dim[1]):
                dist[start[1]+y][start[0]+x] = i
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
            corners = pymzn.minizinc('PWP_CP_implied.mzn', directory + filename)#, output_mode='item')
            corners = corners[0]
            print(corners)
            print(corners['place'])
            now = time.time() - start
            print('time',start, now)
            writeSolutions(txtname, w, h, nPres, dims, corners['place'], now, dir, 'CP')
            dist = make_dist(w,h,corners['place'])
            printPaper(w,h,corners['place'],dir)

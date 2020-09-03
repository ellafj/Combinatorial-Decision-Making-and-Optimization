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
    #list = [ '9x9.dzn','10x10.dzn','11x11.dzn','12x12.dzn', '13x13.dzn', '14x14.dzn', '15x15.dzn', '16x16.dzn', '17x17.dzn']
    list = [ '18x18.dzn','19x19.dzn','20x20.dzn']
    for filename in os.listdir(directory):
        #if filename != '.DS_Store':
        if filename in list:
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
            printPaper(w,h,dist,dir)

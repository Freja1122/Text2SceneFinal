import os
import sys

root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import math
from constants import *

'''
positive mean:
0=<x<=500
0=<y<=400
'''
'''
grid_id = grid_num
'''

canvas_max_x, min_x = 500, 0
canvas_max_y, min_y = 400, 0

grid_num_x = 28
grid_num_y = 28


class Grid_x:
    def __init__(self,
                 x_max=canvas_max_x,
                 x_min=0,
                 y_max=canvas_max_y,
                 y_min=0,
                 grid_num_x=grid_num_x,
                 grid_num_y=grid_num_y,
                 positive_xy=True,
                 sound=False):
        self.name = 'gridx'
        self.y_max = y_max
        self.x_max = x_max
        self.y_min = y_min
        self.x_min = x_min
        self.positive_xy = positive_xy
        if self.positive_xy:
            self.y_min = 0
            self.x_min = 0
            self.y_max = canvas_max_y
            self.x_max = canvas_max_x
        self.x_grid_num = grid_num_x
        self.y_grid_num = grid_num_y
        self.grid_size_y = (self.y_max - self.y_min) / (grid_num_x - 1)
        self.grid_size_x = (self.x_max - self.x_min) / (grid_num_y - 1)
        self.sound = sound
        self.max_grid_id = self.get_grid_id_from_xy(self.x_max, self.y_max)
        print('\ngrid data:')
        print('new grid done:')
        print('y_max', self.y_max)
        print('y_min', self.y_min)
        print('x_max', self.x_max)
        print('x_min', self.x_min)
        print('y_grid_num:', self.y_grid_num)
        print('x_grid_num:', self.x_grid_num)
        print('grid_size_y:', self.grid_size_y)
        print('grid_size_x:', self.grid_size_x)
        print('total_grid_num',self.max_grid_id)

    """
    if x, y or grid_id is out of range, give it the bounding value
    """

    def get_correct_x_y(self, x, y):
        if y < self.y_min:
            y = self.y_min
        if x < self.x_min:
            x = self.x_min
        if y > self.y_max:
            y = self.y_max
        if x > self.x_max:
            x = self.x_max
        return x, y

    def get_grid_id_from_xy(self, x, y):
        x, y = self.get_correct_x_y(x, y)
        x_grid = int(x / self.grid_size_x)
        y_grid = int(y / self.grid_size_y)
        return y_grid * self.x_grid_num + x_grid

    def get_center_pos_from_id(self, grid_id):
        x_grid = grid_id % self.x_grid_num
        y_grid = grid_id // self.x_grid_num
        x = int((float(x_grid) + 0.5) * self.grid_size_x)
        y = int((float(y_grid) + 0.5) * self.grid_size_y)
        return x, y


grid_dict = {
    "gridx": Grid_x}

if __name__ == '__main__':
    grid_x = Grid_x(positive_xy=True)
    # grid_x_y = Grid_x_y(positive_xy=True)
    xy = [0,0]
    print()
    print('xy: ', xy)
    grid_id = grid_x.get_grid_id_from_xy(xy[0], xy[1])
    print('grid_id: ', grid_id)
    pos = grid_x.get_center_pos_from_id(0)
    print('pos: ', pos)
    # pickle_file = '_pickle_data_all/output_all.pkl'
    # with open(os.path.join(root_path, pickle_file), 'rb') as f:
    #     output_all = pickle.load(f)
    #     print('output_all[0]', output_all[0])
    # x_list, y_list = get_max_x_y(output_all)

    # grid_x = Grid_x(positive_xy=True)
    # grid_x_y = Grid_x_y(positive_xy=True)
    # xy = [500, 400]
    # print()
    # print('xy: ', xy)
    # grid_id = grid_x_y.get_grid_from_pos(xy[0], xy[1])
    # geid_id_x, grid_id_y = grid_x_y.get_x_y_grid_from_pos(xy[0], xy[1])
    # print('grid_id: ', geid_id_x, grid_id_y)
    # x, y = grid_x_y.get_center_pos_from_x_y_grid_num(geid_id_x, grid_id_y)
    # print('x,y: ', x, y)

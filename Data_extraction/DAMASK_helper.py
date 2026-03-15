#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:32:16 2019

@author: vinit
"""

import sys
sys.path.append('/home/damaskAscii/')
#import damaskAscii
from matplotlib import pyplot as plt
import numpy as np, pandas as pd

def readin_microstructure(filename):
    in_texture_section = False
    in_geometry_section = False
    orientations = []
    field = []
    for line in open(filename):
        if in_texture_section:
            if 'phi1' in line and 'Phi' in line and 'phi2' in line:
                orientations.append([float(x) for x in line.split()[2:7:2]])
        if in_geometry_section:
            for i_ in line.split():
                i = int(i_)
                field.append([i] + [orientations[i-1][x] for x in range(3)])
        if '<texture>' in line:
            in_texture_section = True
        if '<!skip>' in line:
            in_geometry_section = True
        if 'grid' in line:
            grid_dime = [int(x) for x in line.split()[2:7:2]]
        if 'microstructures' in line:
            num_material_states = int(line.split()[1])
    grid_dime.reverse()
    #field_ = np.array(field).reshape(grid_dime+[4])
    field_ = np.array(field)
    return field_, num_material_states

def readin_grid(filename):
    in_geometry_section = False
    field = []
    for line in open(filename):
        if in_geometry_section:
            for i_ in line.split():
                i = int(i_)
                field.append(i)
        if '<!skip>' in line:
            in_geometry_section = True
        if 'grid' in line:
            grid_dime = [int(x) for x in line.split()[2:7:2]]
        if 'microstructures' in line:
            num_material_states = int(line.split()[1])
    grid_dime.reverse()
    field_ = np.array(field).reshape(grid_dime)
    return field_, num_material_states

def readin_seeds(filename):
    in_seeds_section = False
    seeds = []
    for line in open(filename):
        if in_seeds_section:
            seeds.append([float(x) for x in line.split()])
        if '1_pos	2_pos	3_pos	1_euler	2_euler	3_euler	microstructure' in line:
            in_seeds_section = True
    seeds_scaled = np.array(seeds)
    seeds_scaled[:, [0, 1, 2]] = 2*seeds_scaled[:, [0, 1, 2]]-1
    seeds_scaled[:, [3, 4, 5]] = seeds_scaled[:, [3, 4, 5]]/180.0-1
    return seeds, seeds_scaled

#def readin_results(filename):
#    results = damaskAscii.readDamaskAscii(filename)
#    return results

def read_DAMASK_results(result_files, res_filter, res_items, filter_ids='ALL'):
    '''
    Below are example inputs and usage
    result_file_template = '/home/vinit/phd/2020/work/one_block_in_a_homogenous_medium/runs/change_euler_angles_2_circles/circle_INDEX/postProc/circle_INDEX_tensionY-2_inc2.txt'
    #generate filelist
    result_files = {}
    for i in range(1, 4):
        result_files.update({i:result_file_template.replace('INDEX', str(i))})
    res_filter = 'elem'
    res_items = {'elem':['elem'],
                 'position':['1_pos', '2_pos', '3_pos'],
                 'stress':['5_p']}
    filter_ids = [128, 303, 1185, 1284]
    #filter_ids = 'ALL'
    R = read_DAMASK_results(result_files, res_filter, res_items, filter_ids)
    '''
    res_keys = []
    for x in res_items:
        res_keys += res_items[x]
    
    fdfs = []
    run_keys = []
    count, total = 0, len(result_files)
    sys.stdout.write(' Reading the results\n')
    for i in result_files:
        result_file = result_files[i]
        with open(result_file) as f:
            headerLen = int(f.readline().split()[0])
        df = pd.read_csv(result_file, sep='\t', header=headerLen).set_index(res_filter, drop=False)
        if filter_ids == 'ALL':
            fdf = df[res_keys]
        else:
            fdf = df.loc[filter_ids][res_keys]
        fdfs.append(fdf)
        run_keys.append(i)
        # print status
        count += 1
        sys.stdout.write('\r')
        sys.stdout.write("[%-40s] %d av %d" % ('='*int(40*count/total), count, total))
        sys.stdout.flush()
    sys.stdout.write('\n Concatening the results')
    R = pd.concat(fdfs, keys = run_keys, names = ['run', res_filter], axis = 0)
    return R

def plt_2D_ms(array):
    edge = int(len(array)**0.5)
    plt.matshow(np.reshape(array, [edge, edge]), fignum=0)
    return

def grid_coords_2D(n_edge, i, plot=True):
    if i >= n_edge**2:
        raise ValueError('i = %d is greater than n_edge**2 - 1' %i)
    x = i%n_edge*1/n_edge + 0.5*1/n_edge
    y = int(i/n_edge)*1/n_edge + 0.5*1/n_edge
    if plot:
        plt.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], '.')
        plt.plot(x, y, 'x')
    return [x, y]

def plot_2D_geom_file(geom_file):
    grid_, num_material_states = readin_microstructure(geom_file)
    grid = grid_[:, :, :, 0]
    grid_dim = grid.shape
    plt.matshow(grid.reshape((grid_dim[0], grid_dim[1])))
    seed_file = geom_file.replace('.geom', '.seeds')
    try:
        seeds, seeds_scaled = readin_seeds(seed_file)
        i = 1
        for seed in seeds:
            x_ = (seed[0] + .01) * grid_dim[0]
            y_ = (seed[1] + .01) * grid_dim[1]
            plt.plot(x_, y_, '.k')
            plt.text(x_, y_ , r'${%d}_s$' %(i), fontsize=10)
            i += 1
    except FileNotFoundError:
        pass
    #xy_bucket = [[] for x in range(num_material_states)]
    #for o in itertools.product(range(grid_dim[0]), range(grid_dim[1]), range(grid_dim[2])):
    #    xy_bucket[int(grid[o])-1].append(o[:2])
    #for m in range(num_material_states):
    #    centroid = np.array(xy_bucket[m]).mean(axis=0) + 1
    #    print(centroid)
    #    plt.text(centroid[1], centroid[0], r'$s_{%d}$' %(m+1), fontsize=10)

'''
ASD_mompix
 - Write magnetic snapshots as pixel maps
Author: Anders Bergman, Uppsala University 2024
'''
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def main():
    '''
    Main routine for ASD_mompix
    '''

    def savefig(data, filename, cmap='seismic'):
        '''
         Function to save a matrix as minimal image
        Credits: https://stackoverflow.com/questions/37809697
        '''
        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(data, cmap=cmap)
        fig.savefig(filename, dpi=data.shape[0])
        plt.close(fig)

    def read_posfile(posfile):
        '''
         Read atom positions from UppASD position file
        '''
        with open(posfile, 'r', encoding="utf-8") as pfile:
            lines = pfile.readlines()
            positions = np.empty([0, 3])
            numbers = []
            for _, line in enumerate(lines):
                line_data = line.rstrip('\n').split()
                if len(line_data) > 0:
                    positions = np.vstack(
                        (positions, np.asarray(line_data[2:5]).astype(np.float64)))
                    numbers = np.append(
                        numbers, np.asarray(
                            line_data[1]).astype(
                            np.int32))
        return positions, numbers

    def read_inpsd(ifile):
        '''
         Read important keywords from UppASD inputfile `inpsd.dat`
        '''
        posfiletype = 'C'
        with open(ifile, 'r', encoding="utf-8") as infile:
            lines = infile.readlines()
            for idx, line in enumerate(lines):
                line_data = line.rstrip('\n').split()
                if len(line_data) > 0:
                    # Find the simulation id
                    if line_data[0] == 'simid':
                        simid = line_data[1]

                    # Find the cell data
                    if line_data[0] == 'cell':
                        cell = []
                        lattice = np.empty([0, 3])
                        line_data = lines[idx + 0].split()
                        cell = np.append(cell, np.asarray(line_data[1:4]))
                        lattice = np.vstack(
                            (lattice, np.asarray(line_data[1:4])))
                        line_data = lines[idx + 1].split()
                        cell = np.append(cell, np.asarray(line_data[0:3]))
                        lattice = np.vstack(
                            (lattice, np.asarray(line_data[0:3])))
                        line_data = lines[idx + 2].split()
                        cell = np.append(cell, np.asarray(line_data[0:3]))
                        lattice = np.vstack(
                            (lattice, np.asarray(line_data[0:3])))
                        lattice = lattice.astype(np.float64)

                    # Find the size of the simulated cell
                    if line_data[0] == 'ncell':
                        ncell_x = float(line_data[1])
                        ncell_y = float(line_data[2])
                        ncell_z = float(line_data[3])
                        mesh = [ncell_x, ncell_y, ncell_z]

                    # Read the name of the position file
                    if line_data[0].strip() == 'posfile':
                        positions, numbers = read_posfile(line_data[1])

                    # Read the type of coordinate representation
                    if line_data[0].strip() == 'posfiletype':
                        posfiletype = line_data[1]

        return lattice, positions, numbers, simid, mesh, posfiletype

    ############################################################
    # Open and read input files
    ############################################################
    ifile = 'inpsd.dat'
    _, _, _, simid, _, _ = read_inpsd(ifile)

    ############################################################
    # Read coordinates and magnetic moments
    ############################################################
    coord = np.genfromtxt('coord.' + simid + '.out')
    mom = np.genfromtxt('restart.' + simid + '.out')

    ############################################################
    # Find unique dimensionality of coordinates
    ############################################################
    uni_x = np.unique(coord[:, 1])
    uni_y = np.unique(coord[:, 2])
    uni_z = np.unique(coord[:, 3])

    ############################################################
    # Project moment to positions
    # - Could be done automatically for square geometries
    # - Converts also non-square cells to a square matrix
    ############################################################
    mom_mat = np.zeros((len(uni_x), len(uni_y), len(uni_z)))
    for i, mom_i in enumerate(mom):
        x_idx = np.int32(np.where(uni_x == coord[i, 1])[0])
        y_idx = np.where(uni_y == coord[i, 2])[0]
        z_idx = np.where(uni_z == coord[i, 3])[0]
        # print(i,x_idx,y_idx,z_idx,mom_i[2])
        mom_mat[x_idx, y_idx, z_idx] = mom_i[6]

    ############################################################
    # Print each individual layer and average
    # - Output file names "mompix.simid.Zxxx.png" where xxx is
    #   the z-layer index
    ############################################################
    for layer, _ in enumerate(uni_z):
        fname = "mompix." + simid + f".Z{layer+1:03d}" + ".png"
        savefig(mom_mat[:, :, layer], fname)

    fname = "mompix." + simid + ".png"
    savefig(np.sum(mom_mat, axis=2) / len(uni_z), fname)


if __name__ == "__main__":
    main()

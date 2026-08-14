import gemmi
import numpy as np
import sys
import argparse

def create_ccp4_map_(density_array, output_map_file, voxel_size):
    """
    Create a CCP4 map in real space using the voxel size in Å
    """
    # Get grid shape
    nx, ny, nz = density_array.shape
    #print(f"Grid shape: {nx, ny, nz}")

    # Compute real box dimensions
    box_x = nx * voxel_size
    box_y = ny * voxel_size
    box_z = nz * voxel_size
    #print(f"Real box size (Å): {box_x, box_y, box_z}")

    # Create grid
    grid = gemmi.FloatGrid(nx, ny, nz)
    grid.set_unit_cell(gemmi.UnitCell(box_x, box_y, box_z, 90, 90, 90))
    grid.spacegroup = gemmi.SpaceGroup("P 1")

    # Compute real-space origin shift so center is at (0,0,0)
    origin_shift = np.array([box_x / 2, box_y / 2, box_z / 2])

    # Fill the grid
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Position in real space
                x = i * voxel_size - origin_shift[0]
                y = j * voxel_size - origin_shift[1]
                z = k * voxel_size - origin_shift[2]
                # No need to convert to fractional coords for gemmi; set_value uses grid indices
                grid.set_value(i, j, k, float(density_array[i, j, k]))

    # Write CCP4 map
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_map_file)

    #print(f"CCP4 map saved to: {output_map_file}")
    #print(f"Unit cell (Å): {grid.unit_cell}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--density_array", help="path to 3D density in numpy array form", type=str, required=True)
    parser.add_argument("--voxel_size", help="voxel size of 3D density in Ångström", type=float, required=True)
    parser.add_argument("--output_map", help="output path to ccp4 map object created", type=str, required=True)

    args = parser.parse_args()
    density_array = args.density_array
    voxel_size =  args.voxel_size
    output_map_file = args.output_map
    create_ccp4_map_(density_array, output_map_file, voxel_size)

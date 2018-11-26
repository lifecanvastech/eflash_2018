"""collect-patches - given a list of points, collect the patches around them

"""

import h5py
import multiprocessing
import numpy as np
from phathom.utils import SharedMemory
import tifffile
import json
import argparse
import glob
import tqdm
from .utils import RollingBuffer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",
                        required=True,
                        help="The glob expression for the stack")
    parser.add_argument("--points",
                        required=True,
                        help="A .json file containing the points in xyz order")
    parser.add_argument("--patch-size",
                        type=int,
                        default=31,
                        help="Size of a patch in pixels (an odd # please)")
    parser.add_argument("--output",
                        required=True,
                        help="The location for an HDF file file containing "
                        "an NxMxM array where N is the number of points and "
                        "M is the patch size and 3 arrays containing the X,"
                             "Y and Z coordinates of each of N points.")
    parser.add_argument("--n-cores",
                        default=12,
                        help="The number of cores to use")
    return parser.parse_args()


def do_plane(filename:str,
             points:np.ndarray,
             shared_memory:SharedMemory,
             offset:int):
    """

    :param filename: name of file to parse
    :param points: an N x 2 array of X, Y points at which to sample
    :param shared_memory: Shared memory block to write into
    :param offset: offset into block
    """
    patch_size = shared_memory.shape[1]
    half_size = patch_size // 2
    plane = np.pad(tifffile.imread(filename), half_size, mode='reflect')
    for idx, (x, y) in enumerate(points):
        x0 = x
        x1 = x + patch_size
        y0 = y
        y1 = y + patch_size
        with shared_memory.txn() as m:
            m[offset + idx] = plane[y0:y1, x0:x1]


def main():
    args = parse_args()
    source_files = sorted(glob.glob(args.source))
    rb = RollingBuffer(source_files, args.n_cores)
    points = np.array(json.load(open(args.points)))
    patch_size = args.patch_size
    half_patch_size = patch_size // 2
    patches_xy, patches_xz, patches_yz = \
        [np.zeros((len(points), patch_size, patch_size)) for _ in range(3)]
    points_out = []
    offset = 0
    for z in tqdm.tqdm(
            range(half_patch_size, len(source_files) - half_patch_size)):
        rb.release(z - half_patch_size)
        pz = points[(points[:, 2] >= z) & (points[:, 2] < z+1)]
        if len(pz) > 0:
            for x, y in pz[:, :-1]:
                x, y = int(x), int(y)
                if x < half_patch_size or \
                    x >= rb.shape[2] - half_patch_size or \
                    y < half_patch_size or \
                    y >= rb.shape[1] - half_patch_size:
                    continue
                patches_xy[offset] = rb[
                    z,
                    y - half_patch_size: y + half_patch_size + 1,
                    x - half_patch_size: x + half_patch_size + 1]
                patches_xz[offset] = rb[
                    z - half_patch_size: z + half_patch_size + 1,
                    y,
                    x - half_patch_size: x + half_patch_size + 1]
                patches_yz[offset] = rb[
                    z - half_patch_size: z + half_patch_size + 1,
                    y - half_patch_size: y + half_patch_size + 1,
                    x]
                points_out.append((z, y, x))
                offset += 1
    patches_xy, patches_xz, patches_yz = \
        [_[:offset] for _ in (patches_xy, patches_xz, patches_yz)]
    points_out = np.array(points_out)
    with h5py.File(args.output, "w") as f:
        old_patches = f.create_dataset("patches_xy", data=patches_xy)
        f.create_dataset("patches_xz", data=patches_xz)
        f.create_dataset("patches_yz", data=patches_yz)
        f.create_dataset("x", data=points_out[:, 0])
        f.create_dataset("y", data=points_out[:, 1])
        f.create_dataset("z", data=points_out[:, 2])
        f["patches"] = old_patches


if __name__ == "__main__":
    main()
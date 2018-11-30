"""collect-patches - given a list of points, collect the patches around them

"""

import h5py
import multiprocessing
import numpy as np
import os
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
                        type=int,
                        help="The number of cores to use")
    parser.add_argument("--n-io-cores",
                        default=12,
                        type=int,
                        help="The number of cores to use during I/O")
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


def do_z(pz, offset, patches_xy, patches_xz, patches_yz, rb, z,
         half_patch_size):
    for x, y in pz[:, :-1]:
        x, y = int(x), int(y)
        with patches_xy.txn() as m:
            m[offset] = rb[
                           z,
                           y - half_patch_size: y + half_patch_size + 1,
                           x - half_patch_size: x + half_patch_size + 1]
        with patches_xz.txn() as m:
            m[offset] = rb[
                           z - half_patch_size: z + half_patch_size + 1,
                           y,
                           x - half_patch_size: x + half_patch_size + 1]
        with patches_yz.txn() as m:
            m[offset] = rb[
                           z - half_patch_size: z + half_patch_size + 1,
                           y - half_patch_size: y + half_patch_size + 1,
                           x]
        offset += 1
    return offset


def main():
    args = parse_args()
    source_files = sorted(glob.glob(args.source))
    rb = RollingBuffer(source_files, args.n_io_cores)
    points = np.array(json.load(open(args.points)))
    patch_size = args.patch_size
    half_patch_size = patch_size // 2
    patches_xy, patches_xz, patches_yz = \
        [SharedMemory((len(points), patch_size, patch_size), rb.dtype)
         for _ in range(3)]
    points_out = []
    offset = 0
    x1 = rb.shape[2] - half_patch_size
    y1 = rb.shape[1] - half_patch_size
    with multiprocessing.Pool(args.n_cores) as pool:
        for z in tqdm.tqdm(
                range(half_patch_size, len(source_files) - half_patch_size)):
            rb.release(z - half_patch_size)
            rb.wait(z + half_patch_size)
            pz = points[(points[:, 2] >= z) & (points[:, 2] < z+1)]
            if len(pz) == 0:
                continue
            mask = np.all(pz[:, 0:2] >= half_patch_size, 1) &\
                   (pz[:, 0] < x1) &\
                   (pz[:, 1] < y1)
            pz = pz[mask]
            if len(pz) == 0:
                continue
            points_out.append(pz)
            if len(pz) < args.n_cores * 10 or args.n_cores == 1:
                offset = do_z(pz, offset,
                              patches_xy, patches_xz, patches_yz, rb, z,
                              half_patch_size)
            else:
                idxs = np.linspace(0, len(pz), args.n_cores+1).astype(int)
                frb = rb.freeze()
                fnargs = [(pz[i0:i1], offset + i0,
                           patches_xy, patches_xz, patches_yz, frb, z,
                           half_patch_size)
                          for i0, i1 in zip(idxs[:-1], idxs[1:])]
                pool.starmap(do_z, fnargs)
                offset += len(pz)

    points_out = np.vstack(points_out)
    with h5py.File(args.output, "w") as f:
        with patches_xy.txn() as m:
            old_patches = f.create_dataset("patches_xy", data=m[:offset])
        with patches_xz.txn() as m:
            f.create_dataset("patches_xz", data=m[:offset])
        with patches_yz.txn() as m:
            f.create_dataset("patches_yz", data=m[:offset])
        f.create_dataset("x", data=points_out[:, 0])
        f.create_dataset("y", data=points_out[:, 1])
        f.create_dataset("z", data=points_out[:, 2])
        f["patches"] = old_patches



if __name__ == "__main__":
    main()
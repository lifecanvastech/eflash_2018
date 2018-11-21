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
    points = np.array(json.load(open(args.points)))
    patch_size = args.patch_size
    shared_memory = SharedMemory((len(points), patch_size, patch_size),
                                 np.uint16)
    points_out = []
    offset = 0
    with multiprocessing.Pool(args.n_cores) as pool:
        futures = []
        for z in range(len(source_files)):
            pz = points[(points[:, 2] >= z) & (points[:, 2] < z+1)]
            if len(pz) > 0:
                points_out.append(pz)
                future = pool.apply_async(
                    do_plane,
                    (source_files[z],
                     pz[:, :2].astype(int),
                     shared_memory,
                     offset)
                )
                futures.append(future)
                offset += len(pz)
        for future in tqdm.tqdm(futures):
            future.get()
    points_out = np.vstack(points_out)
    with h5py.File(args.output, "w") as f:
        with shared_memory.txn() as m:
            f.create_dataset("patches", data=m)
            f.create_dataset("x", data=points_out[:, 0])
            f.create_dataset("y", data=points_out[:, 1])
            f.create_dataset("z", data=points_out[:, 2])

if __name__ == "__main__":
    main()
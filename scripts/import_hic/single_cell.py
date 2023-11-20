import multiprocessing
import os
import os.path as osp

from import_contactmap_straw import import_contactmap_straw


def single_cell_import():
    dir = '/home/erschultz/sequences_to_contact_maps'
    # dir = '/project2/depablo/erschultz'
    dataset='single_cell_nagano_2017/samples'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)

    chromosome='10'
    start_mb=0
    resolution=50000
    m=2600
    start = start_mb * 10**6
    end = start + (m-1) * resolution

    # set up for multiprocessing
    mapping = []
    for file in os.listdir(data_folder):
        sample_folder = osp.join(data_folder, file)
        filename = osp.join(sample_folder, 'adj.hic')
        mapping.append((sample_folder, filename, chromosome, start, end, resolution))

    # print(mapping)
    with multiprocessing.Pool(15) as p:
        p.starmap(import_contactmap_straw, mapping)

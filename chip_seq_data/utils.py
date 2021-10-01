import os.path as osp
import pandas as pd

#Autologous chroms
CHROMS = [str(ele) for ele in range(1,23)]
#Sex chrom: X only
CHROMS.append("X")

CHROM_LENGTHS = {'1': 248956422,
                '2': 242193529,
                '3': 198295559,
                '4': 198295559,
                '5': 181538259,
                '6': 170805979,
                '7': 159345973,
                '8': 145138636,
                '9': 138394717,
                '10': 133797422,
                '11': 135086622,
                '12': 133275309,
                '13': 114364328,
                '14': 107043718,
                '15': 101991189,
                '16': 90338345,
                '17': 83257441,
                '18': 80373285,
                '19': 58617616,
                '20': 64444167,
                '21': 46709983,
                '22': 50818468,
                'X': 156040895}

def get_names(dir, files):
    metadata_file = osp.join(dir, 'metadata.tsv')
    if osp.exists(metadata_file):
        metadata = pd.read_csv(metadata_file, sep = '\t')
    else:
        raise Exception('metadata.tsv does not exist')

    names = []
    for file in files:
        accession = osp.split(file)[-1].split('.')[0]
        target = metadata[metadata['File accession'] == accession]['Experiment target'].item()
        name = target.split('-')[0]
        names.append(name)

    names_set = set(names)
    if len(names) != len(names_set):
        print('Warning: duplicate names')

    return names

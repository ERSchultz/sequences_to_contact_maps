from neural_net_utils.networks import *
from neural_net_utils.utils import *

def main():
    sample = 'sample1000'
    dataPath = os.path.join('dataset_04_06_21', sample)
    y = np.load(dataPath + '/y.npy')
    plotExpectedDist(y, sample + '_y_exp.png')

    y_diag = np.load(dataPath + '/y_diag_norm.npy')
    plotExpectedDist(y, sample + '_y_diag_exp.png'))


    modeldir = 'models/TODO'


if __name__ == '__main__':
    main()

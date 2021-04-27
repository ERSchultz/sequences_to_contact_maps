from neural_net_utils.networks import *
from neural_net_utils.utils import *
from neural_net_utils.dataset_classes import Sequences2Contacts

def freqDistributionPlots(dataFolder, n = 1024):
    chi = np.load(os.path.join(dataFolder, 'chis.npy'))
    k = len(chi)

    # freq distribution plots
    for diag in [True, False]:
        print(diag)
        freq_arr = getFrequencies(dataFolder, diag, n, k, chi)
        for split in [None, 'type', 'psi']:
            print(split)
            plotFrequenciesSampleSubplot(freq_arr, dataFolder, diag, k, split)
            plotFrequenciesSubplot(freq_arr, dataFolder, diag, k, sampleid = 1, split = split)

def freqStatisticsPlots(dataFolder):
    # freq statistics plots
    for diag in [True, False]:
        for stat in ['mean', 'var']:
            ofile = os.path.join(dataFolder, "freq_stat_{}_diag_{}.png".format(stat, diag))
            plotDistStats(dataFolder, diag, ofile, stat = stat)


def contactPlots(dataFolder):
    in_paths = sorted(make_dataset(dataFolder))
    for path in in_paths:
        # y = np.load(os.path.join(path, 'y.npy'))
        # plotContactMap(y, os.path.join(path, 'y.png'), title = 'pre normalization', vmax = 'mean')
        #
        # y_diag_norm = np.load(os.path.join(path, 'y_diag_norm.npy'))
        # plotContactMap(y_diag_norm, os.path.join(path, 'y_diag_norm.png'), title = 'diag normalization', vmax = 'mean')

        y_prcnt_norm = np.load(os.path.join(path, 'y_prcnt_norm.npy'))
        plotContactMap(y_prcnt_norm, os.path.join(path, 'y_prcnt_norm.png'), title = 'prcnt normalization', vmax = 'mean')

def plot_predictions(dataFolder, modelName):
    seq2ContactData = Sequences2Contacts(dataFolder, toxx = True,
                                        names = True, min_sample = 0)
    dataloader, _, _ = getDataLoaders(seq2ContactData, batch_size = 1,
                                            num_workers = 0, seed = 42,
                                            split = [1,0,0], shuffle = False)


    if modelName is not None:
        modeldir = os.path.join('models', modelName)
        model = UNet(nf_in = 2, nf_out = 1, nf = 8, out_act = nn.Sigmoid())
        save_dict = torch.load(modeldir, map_location = 'cpu')
        model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
    vmax = 'mean'
    for (x, y, name) in dataloader:
        print(name)
        name = name[0].split('/')[1] # TODO use os.path.split
        yhat = model(x)
        plotContactMap(y, os.path.join(dataFolder, name, '_y.png'), vmax = vmax, title = 'Y')
        plotContactMap(yhat.detach().numpy(), os.path.join(dataFolder, name, '_yhat.png'), vmax = vmax, title = 'Y hat')
        ydif = yhat.detach().numpy() - y.numpy()
        plotContactMap(ydif, os.path.join(dataFolder, name, '_ydif.png'), vmax = vmax, title = 'difference')
        loss = F.mse_loss(yhat, y)
        print(loss)

def main():
    # freqDistributionPlots('dataset_04_18_21')
    # freqStatisticsPlots('dataset_04_18_21')
    contactPlots('dataset_04_18_21')
    # plot_predictions('dataset_04_18_21', None)


if __name__ == '__main__':
    main()

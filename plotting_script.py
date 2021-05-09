from neural_net_utils.networks import *
from neural_net_utils.utils import *
from neural_net_utils.plotting_functions import *
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
        print(path)
        y = np.load(os.path.join(path, 'y.npy'))
        plotContactMap(y, os.path.join(path, 'y.png'), title = 'pre normalization', vmax = 'mean')

        y_diag_norm = np.load(os.path.join(path, 'y_diag.npy'))
        plotContactMap(y_diag_norm, os.path.join(path, 'y_diag.png'), title = 'diag normalization', vmax = 'max')

        y_prcnt_norm = np.load(os.path.join(path, 'y_prcnt.npy'))
        plotContactMap(y_prcnt_norm, os.path.join(path, 'y_prcnt.png'), title = 'prcnt normalization', vmax = 'max', prcnt = True)

def main():
    opt = argparseSetup()
    opt.mode = None
    # opt.mode = 'debugging'
    # overwrites if testing locally
    if opt.mode == 'debugging':
        opt.data_folder = 'dataset_04_18_21'
        opt.loss = 'cross_entropy'
        opt.y_preprocessing = 'prcnt'
        opt.y_norm = None
        opt.n_epochs = 15
        opt.milestones = [5,10]
        opt.lr = 0.1
        opt.toxx = True

    print(opt)
    opt.model_type = 'UNet'
    if opt.model_type == 'UNet':
        opt.ofile = "UNet_nEpochs{}_nf{}_lr{}_milestones{}_yPreprocessing{}_yNorm{}".format(opt.n_epochs, opt.nf, opt.lr, list2str(opt.milestones), opt.y_preprocessing, opt.y_norm)
        if opt.loss == 'cross_entropy':
            assert opt.y_preprocessing == 'prcnt', 'must use percentile normalization with cross entropy'

            opt.criterion = F.cross_entropy

            # change default params
            opt.y_reshape = False
            opt.ydtype = torch.int64
            model = UNet(nf_in = 2, nf_out = 10, nf = opt.nf, out_act = None) # activation combined into loss
        elif opt.loss == 'mse':
            opt.criterion = F.mse_loss
            model = UNet(nf_in = 2, nf_out = 1, nf = opt.nf, out_act = nn.Sigmoid())
        else:
            print('Invalid loss: {}'.format(opt.loss))
    elif opt.model_type == 'DeepC':
        opt.ofile = "DeepC_nEpochs{}_nf{}_lr{}_milestones{}_yPreprocessing{}_kernelW{}_hiddenSize{}_dilation{}_hiddenSize_{}".format(opt.n_epochs, opt.nf, opt.lr, list2str(opt.milestones), opt.y_preprocessing, list2str(opt.kernel_w_list), list2str(opt.hidden_sizes_list), list2str(opt.dilation_list), opt.hidden_size_dilation)
        if opt.loss == 'mse':
            opt.criterion = F.mse_loss
        else:
            print('Invalid loss: {}'.format(opt.loss))
    else:
        print('Invalid model type: {}'.format(opt.model_type))
        # TODO

    seq2ContactData = Sequences2Contacts(opt.data_folder, opt.toxx, opt.y_preprocessing,
                                        opt.y_norm, opt.x_reshape, opt.ydtype,
                                        opt.y_reshape, opt.crop)
    train_dataloader, val_dataloader, test_dataloader = getDataLoaders(seq2ContactData, opt, names = True, max = True)
    model_name = os.path.join(opt.ofile_folder, opt.ofile + '.pt')
    if os.path.exists(model_name):
        save_dict = torch.load(model_name, map_location=torch.device('cpu'))
        model.load_state_dict(save_dict['model_state_dict'])
        print('Model is loaded: {}'.format(model_name))
    else:
        print('Model does not exist: {}'.format(model_name))

    # comparePCA(val_dataloader, model, opt)
    if opt.mode == 'debugging':
        imageSubPath = os.path.join('images', 'test')
    else:
        imageSubPath = os.path.join('images', opt.ofile)
    if not os.path.exists(imageSubPath):
        os.mkdir(imageSubPath, mode = 0o755)
    imagePath = os.path.join(imageSubPath, 'distance_pearson.png')
    # plotDistanceStratifiedPearsonCorrelation(val_dataloader, model, imagePath, opt)
    print()

    imagePath = os.path.join(imageSubPath, 'per_class_acc.png')
    # plotPerClassAccuracy(val_dataloader, opt, imagePath)
    print()

    plotPredictions(val_dataloader, opt)
    print('\n'*3)

    # freqDistributionPlots('dataset_04_18_21')
    # freqStatisticsPlots('dataset_04_18_21')
    # contactPlots('dataset_04_18_21')


if __name__ == '__main__':
    main()

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
        print(path)
        y = np.load(os.path.join(path, 'y.npy'))
        plotContactMap(y, os.path.join(path, 'y.png'), title = 'pre normalization', vmax = 'mean')

        y_diag_norm = np.load(os.path.join(path, 'y_diag_norm.npy'))
        plotContactMap(y_diag_norm, os.path.join(path, 'y_diag_norm.png'), title = 'diag normalization', vmax = 'max')

        y_prcnt_norm = np.load(os.path.join(path, 'y_prcnt_norm.npy'))
        plotContactMap(y_prcnt_norm, os.path.join(path, 'y_prcnt_norm.png'), title = 'prcnt normalization', vmax = 'max', prcnt = True)

def plot_predictions(opt):
    seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = opt.toxx, y_preprocessing = opt.y_preprocessing,
                                        y_norm = opt.y_norm, names = True, y_reshape = opt.reshape)
    _, val_dataloader, _ = getDataLoaders(seq2ContactData, batch_size = opt.batch_size,
                                            num_workers = opt.num_workers, seed = opt.seed)

    subFolder = opt.model_name[:-3]
    imagePath = os.path.join('images', subFolder)
    if not os.path.exists(imagePath):
        os.mkdir(imagePath, mode = 0o755)

    val_n = len(val_dataloader)
    loss_arr = np.zeros(val_n)
    acc_arr = np.zeros(val_n)
    acc_c_arr = np.zeros((val_n, opt.classes))
    freq_c_arr = np.zeros((val_n, opt.classes))
    for i, (x, y, path, max) in enumerate(val_dataloader):
        path = path[0]
        print(path)
        max = float(max)
        subpath = os.path.join(path, subFolder)
        if not os.path.exists(subpath):
            os.mkdir(subpath, mode = 0o755)

        yhat = opt.model(x)
        loss = criterion(yhat, y).item()
        loss_arr[i] = loss
        y = y.numpy()

        yhat = yhat.detach().numpy()

        if opt.prcnt:
            yhat = np.argmax(yhat, axis = 1)
            if opt.plot:
                plotContactMap(yhat, os.path.join(subpath, 'yhat.png'), vmax = 'max', prcnt = True, title = 'Y hat')
            acc = np.sum((yhat == y)) / y.size
            acc_arr[i] = acc
            for c in range(opt.classes):
                denom = np.sum(y == c)
                freq_c_arr[i, c] = denom / y.size
                num = np.sum(np.logical_and((y == c), (yhat == y)))
                acc = num / denom
                acc_c_arr[i, c] = acc
        else:
            yhat = yhat * max
            if opt.plot:
                plotContactMap(yhat, os.path.join(subpath, 'yhat.png'), vmax = 'max', prcnt = False, title = 'Y hat')

            # plot prcnt
            prcntDist_path = os.path.join(opt.data_folder, 'prcntDist.npy')
            prcntDist = np.load(prcntDist_path)
            yhat_prcnt = percentile_normalize(yhat, prcntDist)

            if opt.plot:
                plotContactMap(yhat_prcnt, os.path.join(subpath, 'yhat_prcnt.png'), vmax = 'max', prcnt = True, title = 'Y hat prcnt')

            ytrue = np.load(os.path.join(path, 'y_prcnt_norm.npy'))
            acc = np.sum(yhat_prcnt == ytrue) / yhat.size
            acc_arr[i] = acc
            for c in range(opt.classes):
                denom = np.sum(ytrue == c)
                freq_c_arr[i, c] = denom / ytrue.size
                num = np.sum(np.logical_and((ytrue == c), (yhat_prcnt == ytrue)))
                acc = num / denom
                acc_c_arr[i, c] = acc

            # plot dif
            ydif = yhat - y
            if opt.plot:
                plotContactMap(ydif, os.path.join(subpath, 'ydif.png'), vmax = 'mean', title = 'difference')

    print('Accuracy: {} +- {}'.format(np.mean(acc_arr), np.std(acc_arr)))
    print('Loss: {} +- {}'.format(np.mean(loss_arr), np.std(loss_arr)))
    print(acc_c_arr)
    print(freq_c_arr)
    plotPerClassAccuracy(acc_c_arr, freq_c_arr, ofile = os.path.join(imagePath, 'per_class_acc.png'))


def main():
    opt = argparseSetup()

    if opt.model_type == 'UNet':
        opt.ofile = "UNet_nEpochs{}_nf{}_lr{}_milestones{}_yPreprocessing{}_yNorm{}".format(opt.n_epochs, opt.nf, opt.lr, list2str(opt.milestones), opt.y_preprocessing, opt.y_norm)
        if opt.loss == 'cross_entropy':
            assert opt.y_preprocessing == 'prcnt', 'must use percentile normalization with cross entropy'
            opt.y_reshape = False
            opt.criterion = F.cross_entropy
            opt.ydtype = torch.int64
            model = UNet(nf_in = 2, nf_out = 10, nf = opt.nf, out_act = None) # activation combined into loss
        elif opt.loss == 'mse':
            opt.criterion = F.mse_loss
            model = UNet(nf_in = 2, nf_out = 1, nf = opt.nf, out_act = nn.Sigmoid())
            opt.y_reshape = True
            opt.ydtype = torch.float32
        else:
            print('Invalid loss: {}'.format(opt.loss))

        seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = opt.toxx, y_preprocessing = opt.y_preprocessing,
                                            y_norm = opt.y_norm, x_reshape = opt.x_reshape, ydtype = opt.ydtype,
                                            y_reshape = opt.y_reshape, crop = opt.crop)
    else:
        print('Invalid model type: {}'.format(opt.model_type))
        # TODO
    model_name = os.path.join(opt.ofile_folder, opt.ofile, '.pt')
    if os.path.exists(model_name):
        saveDict = torch.load(model_name, map_location=torch.device('cpu'))
        model.load_state_dict(save_dict['model_state_dict'])
        print('Model is loaded.')
    else:
        print('Model does not exist: {}'.format(model_name))


    train_dataloader, val_dataloader, test_dataloader = getDataLoaders(dataset, opt)

    comparePCA(val_dataloader, model, opt)
    imageSubPath = os.path.join('images', opt.ofile)
    if not os.path.exists(imageSubPath):
        os.mkdir(imageSubPath, mode = 0o755)
    imagePath = os.path.join(imageSubPath, 'distance_pearson.png')
    plotDistanceStratifiedPearsonCorrelation(val_dataloader, model, imagePath, opt)

    # freqDistributionPlots('dataset_04_18_21')
    # freqStatisticsPlots('dataset_04_18_21')
    # contactPlots('dataset_04_18_21')
    # plot_predictions(opt)
    # plot_predictions('dataset_04_18_21', 'UNet_nEpochs15_nf8_lr0.1_milestones5-10_yNormprcnt.pt', 'UNet', 'prcnt')


if __name__ == '__main__':
    main()

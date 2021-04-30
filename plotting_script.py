from neural_net_utils.networks import *
from neural_net_utils.utils import *
from neural_net_utils.dataset_classes import Sequences2Contacts
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA


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


def setupParser():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--data_folder', type=str, default='dataset_04_18_21', help='Location of input data')
    parser.add_argument('--model_name', type=str, default='UNet_nEpochs15_nf8_lr0.1_milestones5-10_yNormdiag.pt', help='name of model to load')
    parser.add_argument('--model_type', type=str, default='UNet', help='type of model')
    parser.add_argument('--y_norm', type=str, default='diag', help='type of y normalization')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of processes to use')
    parser.add_argument('--k', type=int, default=2, help='Number of epigenetic marks')
    parser.add_argument('--n', type=int, default=1024, help='Number of particles')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default: 42')
    parser.add_argument('--plot', type=str2bool, default=False, help='whether or not to plot')
    parser.add_argument('--classes', type=int, default=10, help='number of classes in percentile normalization')

    opt = parser.parse_args()

    if opt.model_type == 'UNet':
        opt.toxx = True
        if opt.y_norm == 'diag':
            opt.model = UNet(nf_in = 2, nf_out = 1, nf = 8, out_act = nn.Sigmoid())
            opt.prcnt = False
            opt.reshape = True
            criterion = F.mse_loss
        elif opt.y_norm == 'prcnt':
            opt.model = UNet(nf_in = 2, nf_out = 10, nf = 8, out_act = None)
            opt.prcnt = True
            opt.reshape = False
            criterion = F.cross_entropy
        else:
            print('Invalid y_norm ({}) for model_type UNet'.format(opt.y_norm))
    else:
        opt.toxx = False
        # TODO

    modeldir = os.path.join('models', opt.model_name)
    save_dict = torch.load(modeldir, map_location = 'cpu')
    opt.model.load_state_dict(save_dict['model_state_dict'])
    opt.model.eval()

    return opt

def plot_predictions(opt):
    seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = opt.toxx, y_norm = opt.y_norm,
                                        names = True, y_reshape = opt.reshape)
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


def comparePCA(opt):
    seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = opt.toxx, y_norm = opt.y_norm,
                                        names = True, y_reshape = opt.reshape)
    _, val_dataloader, _ = getDataLoaders(seq2ContactData, batch_size = opt.batch_size,
                                            num_workers = opt.num_workers, seed = opt.seed)


    val_n = len(val_dataloader)
    acc_arr = np.zeros(val_n)
    rho_arr = np.zeros(val_n)
    p_arr = np.zeros(val_n)
    pca = PCA()
    for i, (x, y, path, max) in enumerate(val_dataloader):
        path = path[0]
        max = float(max)

        yhat = opt.model(x)
        y = y.numpy().reshape((opt.n,opt.n))
        yhat = yhat.detach().numpy().reshape((opt.n,opt.n))

        if opt.prcnt:
            yhat = np.argmax(yhat, axis = 1)
        result_y = pca.fit(y)
        comp1_y = pca.components_[0]
        sign1_y = np.sign(comp1_y)

        result_yhat = pca.fit(yhat)
        comp1_yhat = pca.components_[0]
        sign1_yhat = np.sign(comp1_yhat)
        acc = np.sum((sign1_yhat == sign1_y)) / sign1_y.size
        acc_arr[i] = acc

        corr, pval = spearmanr(comp1_yhat, comp1_y)
        rho_arr[i] = corr

        corr, pval = pearsonr(comp1_yhat, comp1_y)
        p_arr[i] = corr

    print(opt.model_name)
    print('Accuracy: {} +- {}'.format(np.mean(acc_arr), np.std(acc_arr)))
    print('Spearman R: {} +- {}'.format(np.mean(rho_arr), np.std(rho_arr)))
    print('Pearson R: {} +- {}'.format(np.mean(p_arr), np.std(p_arr)))



def main():
    opt = setupParser()
    # freqDistributionPlots('dataset_04_18_21')
    # freqStatisticsPlots('dataset_04_18_21')
    # contactPlots('dataset_04_18_21')
    # plot_predictions(opt)
    comparePCA(opt)
    # plot_predictions('dataset_04_18_21', 'UNet_nEpochs15_nf8_lr0.1_milestones5-10_yNormprcnt.pt', 'UNet', 'prcnt')


if __name__ == '__main__':
    main()

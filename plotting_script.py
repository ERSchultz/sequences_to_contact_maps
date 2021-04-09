from neural_net_utils.networks import *
from neural_net_utils.utils import *
from neural_net_utils.dataset_classes import Sequences2Contacts

def expected_plots():
    sample = 'sample1000'
    dataPath = os.path.join('dataset_04_06_21', sample)
    y = np.load(dataPath + '/y.npy')
    plotExpectedDist(y, sample + '_y_exp.png', title = 'pre normalization', y_scale = 'log')

    y_diag = np.load(dataPath + '/y_diag_norm.npy')
    plotExpectedDist(y_diag, sample + '_y_diag_exp.png', title = 'post normalization')

def main():
    seq2ContactData = Sequences2Contacts('dataset_04_06_21', n = 1024, k = 2, toxx = True)
    dataloader, _, _ = getDataLoaders(seq2ContactData, batch_size = 1,
                                            num_workers = 0, seed = 42,
                                            split = [1,0,0], shuffle = False)


    modeldir = 'models/UNet_nEpochs15_nf8_lr0.01_milestones5-10.pt'
    model = UNet(nf_in = 2, nf_out = 1, nf = 8, out_act = nn.Sigmoid())
    save_dict = torch.load(modeldir, map_location = 'cpu')
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()
    for (x, y, name) in dataloader:
        name = name[0].split('/')[1]
        print(name)
        yhat = model(x)
        plotContactMap(yhat.detach().numpy(), name + '_yhat.png', divide_by_mean = False)
        plotContactMap(y.numpy(), name + '_y.png', divide_by_mean = False)
        loss = F.mse_loss(yhat, y)


if __name__ == '__main__':
    main()

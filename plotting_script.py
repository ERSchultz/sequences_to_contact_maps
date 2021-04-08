from neural_net_utils.networks import *
from utils import *

def main():
    dir = 'models/model_4_8_21.pt'
    model = SimpleEpiNet(1024, 1, 2)
    plotModelFromDir(dir, model, 'model1_train_loss.png')

    dir = 'models/UNET1.pt'
    k=2
    model = UNet(nf_in = 2, nf_out = 1, nf = 4, out_act = nn.Sigmoid())
    plotModelFromDir(dir, model, 'UNET_train_loss.png')

if __name__ == '__main__':
    main()

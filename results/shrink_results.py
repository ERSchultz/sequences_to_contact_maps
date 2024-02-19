import torch
import os.path as osp
import os

def shrink():
    dir = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy'
    assert osp.exists(dir), f'{dir} does not exist'
    print(dir)
    for i in [662, 668, 673]:
        i_dir = osp.join(dir, str(i))
        print(i_dir)
        model_dir = osp.join(i_dir, 'model.pt')
        if osp.exists(model_dir):
            save_dict = torch.load(model_dir, map_location='cpu')
            del save_dict['optimizer_state_dict']
            torch.save(save_dict, osp.join(i_dir, 'model.pt'))

if __name__ == '__main__':
    shrink()

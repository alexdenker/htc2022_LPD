import os 

import argparse

from odl.tomo.backends import ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE

if ASTRA_CUDA_AVAILABLE:
    IMPL = "astra_cuda"
else:
    IMPL = "astra_cpu"


import torch 
from torchvision.transforms.functional import rotate

from PIL import Image
from scipy.io import loadmat
import mat73
import yaml
from skimage.filters import threshold_otsu
import numpy as np 

import odl 
from odl.contrib.torch import OperatorModule
from scipy.ndimage.morphology import binary_closing

from create_ray_transform import get_ray_trafo
from model import SegmentationPrimalDualNet


parser = argparse.ArgumentParser(description='Apply CT-reconstructor to every image in a directory.')

parser.add_argument('input_files')
parser.add_argument('output_files')
parser.add_argument('step', type=int)

step_to_angular_idx = {
    1: 181,
    2: 161,
    3: 141,
    4: 121,
    5: 101,
    6: 81,
    7: 61
}

step_to_angular_range = {
    1: 90,
    2: 80,
    3: 70,
    4: 60,
    5: 50,
    6: 40,
    7: 30
}

def load_image(path):
    try: 
        # for matplab 5.0 files
        ta_sinogram = loadmat(path, struct_as_record=False, simplify_cells=True)
    except:
        print("File could not be loaded using scipy.io.loadmat. Try mat73 instead.")
        ta_sinogram = mat73.loadmat(path)

    sinogram = ta_sinogram["CtDataLimited"]["sinogram"]
    angles = ta_sinogram["CtDataLimited"]["parameters"]["angles"]

    return sinogram, angles


def main(inputFolder, outputFolder, categoryNbr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #### load model ####

    # 1. load ray_trafo  
    ray_trafo = get_ray_trafo(0, step_to_angular_idx[categoryNbr], impl=IMPL)
    fbp_op = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo, frequency_scaling=0.75, filter_type='Hann')

    ray_trafo_torch = OperatorModule(ray_trafo)
    fbp_op_torch = OperatorModule(fbp_op)
            
    ### create file structure
    version_num = 0
    path_parts = ["network_weights", "range=" + str(step_to_angular_range[categoryNbr]), "version_" + str(version_num)]
    log_dir = os.path.join(*path_parts)

    with open(os.path.join(log_dir, "hparams.yml"), 'r') as h_file:
        hparams = yaml.safe_load(h_file)
    
    model = SegmentationPrimalDualNet(n_iter=hparams["n_iter"], 
                                op=ray_trafo_torch, 
                                op_adj=fbp_op_torch, 
                                n_primal=hparams["n_primal"], 
                                n_dual=hparams["n_dual"], 
                                use_sigmoid=hparams["use_sigmoid"], 
                                n_layer=hparams["n_layer"],
                                internal_ch=hparams["internal_ch"], 
                                kernel_size=hparams["kernel_size"], 
                                batch_norm=hparams["batch_norm"],
                                normalize_sinogram=hparams["normalize_sinogram"])

    model.load_state_dict(torch.load(os.path.join(log_dir, "model.pt"),  map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    for f in os.listdir(inputFolder):
        print(f)
        sinogram, angles = load_image(os.path.join(inputFolder, f))
        print(angles)
        y = torch.from_numpy(sinogram).unsqueeze(0).unsqueeze(0).float().to(device)
        
        """
        with torch.no_grad():
            x_hat = torch.sigmoid(model(y))
            x_hat = rotate(x_hat, angles[0])

            xs = []
            for phi in np.linspace(0, 180, 50):
                x_rotate = rotate(x_hat, phi)

                y = ray_trafo_torch(x_rotate)
                y_noise = y + 0.0*hparams["rel_noise"]*torch.mean(torch.abs(y))*torch.randn(y.shape).to(device)

                x_out = rotate(torch.sigmoid(model(y_noise)), -phi)

                xs.append(torch.round(x_out).cpu().numpy()[0,0,:,:])

            x_seg = np.round(np.mean(np.asarray(xs), axis=0))
            print(x_seg.shape)
        """


        with torch.no_grad():
            x_hat = torch.sigmoid(model(y))
            filter = threshold_otsu(x_hat.cpu().numpy()[0,:,:,:])

        print(filter)
        x_seg = rotate(x_hat, angles[0]).cpu().detach().numpy()[0,0,:,:]
        x_seg[x_seg < filter] = 0.
        x_seg[x_seg >= filter] = 1.

        mask = np.ones([7,7])
        mask[0,[0,-1]]=0
        mask[-1,[0,-1]]=0
        x_seg = binary_closing(x_seg, structure=mask)

        im = Image.fromarray(x_seg*255.).convert("L")

        os.makedirs(outputFolder, exist_ok=True)
        im.save(os.path.join(outputFolder,f.split(".")[0] + ".PNG"))

    return 0 


if __name__ == "__main__":

    args = parser.parse_args()


    main(args.input_files, args.output_files, args.step)
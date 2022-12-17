import os
import numpy as np
import torch

from psbody.mesh import mesh

from TailorNet_master.models.tailornet_model import get_best_runner as get_tn_runner
from TailorNet_master.models.smpl4garment import SMPL4Garment
from TailorNet_master.utils.rotation import normalize_y_rotation

from TailorNet_master.dataset.canonical_pose_dataset import get_style, get_shape
from TailorNet_master.visualization.vis_utils import get_specific_pose, get_specific_style_old_tshirt
from TailorNet_master.visualization.vis_utils import get_specific_shape, get_amass_sequence_thetas
from TailorNet_master.utils.interpenetration import remove_interpenetration_fast,get_fn_vn
from TailorNet_master.utils.smpl_paths import SmplPaths
import pickle as pkl
OUT_PATH = "output"


def get_single_frame_inputs(garment_class, gender,betas_from_file):
    """Prepare some individual frame inputs."""
    betas = [
        betas_from_file
       
    ]
   
    if garment_class == 'old-t-shirt':
        gammas = [
           
            get_specific_style_old_tshirt('big_shortsleeve'),
        ]
    else:
        gammas = [
            get_style('000', garment_class=garment_class, gender=gender),
            
        ]
    thetas = [
        get_specific_pose(0),
        
    ]
    return thetas, betas, gammas


def get_sequence_inputs(garment_class, gender):
    """Prepare sequence inputs."""
    beta = get_specific_shape('somethin')
    if garment_class == 'old-t-shirt':
        gamma = get_specific_style_old_tshirt('big_longsleeve')
    else:
        gamma = get_style('000', gender=gender, garment_class=garment_class)

    
    thetas = get_amass_sequence_thetas('05_02')[::2]

    betas = np.tile(beta[None, :], [thetas.shape[0], 1])
    gammas = np.tile(gamma[None, :], [thetas.shape[0], 1])
    return thetas, betas, gammas


def run_tailornet(uniqueId,garmentClass):
    gender = 'female'
    garment_class = garmentClass
    betas_from_file=np.loadtxt(uniqueId+'.txt',dtype=float)
    thetas, betas, gammas = get_single_frame_inputs(garment_class, gender,betas_from_file)
    
    tn_runner = get_tn_runner(gender=gender, garment_class=garment_class)
    
    smpl = SMPL4Garment(gender=gender)

    
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)
   
    for i, (theta, beta, gamma) in enumerate(zip(thetas, betas, gammas)):
        print(i, len(thetas))
       
        theta_normalized = normalize_y_rotation(theta)
        with torch.no_grad():
            pred_verts_d = tn_runner.forward(
                thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(beta[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
            )[0].cpu().numpy()

       
        body, pred_gar = smpl.run(beta=beta, theta=theta, garment_class=garment_class, garment_d=pred_verts_d)
        
        vt, ft = SmplPaths.get_vt_ft_hres('TailorNet_master/dataset/smpl_vt_ft.pkl')
        body.vt=vt
        body.ft=ft
        
        pred_gar = remove_interpenetration_fast(pred_gar, body)
        if garmentClass=='t-shirt':
            garmentTexturePoints = pkl.load(open('TailorNet_master/dataset/vt_ft_file.pkl', 'rb'), encoding='latin1')
            pred_gar.vt=garmentTexturePoints['shirts']['vt']
            pred_gar.ft=garmentTexturePoints['shirts']['ft']
        else:
            garmentTexturePoints = pkl.load(open('TailorNet_master/dataset/vt_ft_file.pkl', 'rb'), encoding='latin1')
            pred_gar.vt=garmentTexturePoints['pants']['vt']
            pred_gar.ft=garmentTexturePoints['pants']['ft']
        
        body.write_obj("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj")
        body.write_mtl("meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".mtl", "body"+"_"+uniqueId, "body"+"_"+uniqueId+'.jpg')
        pred_gar.set_texture_image('Textures/'+"texture_"+"-MeTj4ktsRoBywjktqn-" + ".jpg")
        pred_gar.write_mtl("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".mtl", garmentClass+"_"+uniqueId, garmentClass+"_"+uniqueId+'.jpg')
        pred_gar.write_obj("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj")







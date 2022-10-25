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
# Set output path where inference results will be stored
OUT_PATH = "output"


def get_single_frame_inputs(garment_class, gender,betas_from_file):
    """Prepare some individual frame inputs."""
    betas = [
        betas_from_file
        # get_specific_shape('tallthin'),
        # get_specific_shape('shortfat'),
        # get_specific_shape('mean'),
        # get_specific_shape('somethin'),
        # get_specific_shape('somefat'),
    ]
    # old t-shirt style parameters are centered around [1.5, 0.5, 1.5, 0.0]
    # whereas all other garments styles are centered around [0, 0, 0, 0]
    if garment_class == 'old-t-shirt':
        gammas = [
            # get_specific_style_old_tshirt('mean'),
            # get_specific_style_old_tshirt('big'),
            # get_specific_style_old_tshirt('small'),
            # get_specific_style_old_tshirt('shortsleeve'),
            get_specific_style_old_tshirt('big_shortsleeve'),
        ]
    else:
        gammas = [
            get_style('000', garment_class=garment_class, gender=gender),
            # get_style('001', garment_class=garment_class, gender=gender),
            # get_style('002', garment_class=garment_class, gender=gender),
            # get_style('003', garment_class=garment_class, gender=gender),
            # get_style('004', garment_class=garment_class, gender=gender),
        ]
    thetas = [
        get_specific_pose(0),
        # get_specific_pose(1),
        # get_specific_pose(2),
        # get_specific_pose(3),
        # get_specific_pose(4),
    ]
    return thetas, betas, gammas


def get_sequence_inputs(garment_class, gender):
    """Prepare sequence inputs."""
    beta = get_specific_shape('somethin')
    if garment_class == 'old-t-shirt':
        gamma = get_specific_style_old_tshirt('big_longsleeve')
    else:
        gamma = get_style('000', gender=gender, garment_class=garment_class)

    # downsample sequence frames by 2
    thetas = get_amass_sequence_thetas('05_02')[::2]

    betas = np.tile(beta[None, :], [thetas.shape[0], 1])
    gammas = np.tile(gamma[None, :], [thetas.shape[0], 1])
    return thetas, betas, gammas


def run_tailornet(uniqueId,garmentClass):
    gender = 'male'
    garment_class = garmentClass
    betas_from_file=np.loadtxt(uniqueId+'.txt',dtype=float)
    thetas, betas, gammas = get_single_frame_inputs(garment_class, gender,betas_from_file)
    # # uncomment the line below to run inference on sequence data
    # thetas, betas, gammas = get_sequence_inputs(garment_class, gender)

    # load model
    tn_runner = get_tn_runner(gender=gender, garment_class=garment_class)
    # from trainer.base_trainer import get_best_runner
    # tn_runner = get_best_runner("/BS/cpatel/work/data/learn_anim/tn_baseline/{}_{}/".format(garment_class, gender))
    smpl = SMPL4Garment(gender=gender)

    # make out directory if doesn't exist
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)
    # run inference
    for i, (theta, beta, gamma) in enumerate(zip(thetas, betas, gammas)):
        print(i, len(thetas))
        # normalize y-rotation to make it front facing
        theta_normalized = normalize_y_rotation(theta)
        with torch.no_grad():
            pred_verts_d = tn_runner.forward(
                thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(beta[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
            )[0].cpu().numpy()

        # get garment from predicted displacements
        body, pred_gar = smpl.run(beta=beta, theta=theta, garment_class=garment_class, garment_d=pred_verts_d)
        # body.vt=np.load('TailorNet_master/dataset/basicModel_vt.npy')
        # body.ft=np.load('TailorNet_master/dataset/basicModel_ft.npy')
        # vt_ft=pkl.load(open('TailorNet_master/dataset/smpl_vt_ft.pkl', 'rb'), encoding='latin1')
        vt, ft = SmplPaths.get_vt_ft_hres('TailorNet_master/dataset/smpl_vt_ft.pkl')
        body.vt=vt
        body.ft=ft
        # print(garmentTexturePoints['shirts'])
        # print(garmentTexturePoints['shirts']['vt'])
        # print(vt.shape)
        # print(ft.shape)
        # print(body.f.shape)
        # print(body.v.shape)
        pred_gar = remove_interpenetration_fast(pred_gar, body)
        if garmentClass=='t-shirt':
            texturedGarment=mesh()
            texturedGarment.load_from_obj('tex_'+garmentClass+'.obj')
            pred_gar.vt=texturedGarment.vt
            pred_gar.ft=texturedGarment.ft
        else:
            garmentTexturePoints = pkl.load(open('TailorNet_master/dataset/vt_ft_file.pkl', 'rb'), encoding='latin1')
            pred_gar.vt=garmentTexturePoints['pants']['vt']
            pred_gar.ft=garmentTexturePoints['pants']['ft']
        # print(pred_gar.vt.shape)
        # print(pred_gar.ft.shape)
        # print(pred_gar.v.shape)
        # print(pred_gar.f.shape)
        body.set_texture_image('skins/skin_1.jpg')
        body.write_obj("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj")
        body.write_mtl("meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".mtl", "body"+"_"+uniqueId, "body"+"_"+uniqueId+'.jpg')
        # pred_gar.set_texture_image('Textures/'+"texture_"+textureId + ".jpg")
        pred_gar.write_mtl("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".mtl", garmentClass+"_"+uniqueId, garmentClass+"_"+uniqueId+'.jpg')
        pred_gar.write_obj("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj")

# def render_images():
#     """Render garment and body using blender."""
#     i = 0
#     while True:
#         body_path = os.path.join(OUT_PATH, "body_{:04d}.ply".format(i))
#         if not os.path.exists(body_path):
#             break
#         body = Mesh(filename=body_path)
#         pred_gar = Mesh(filename=os.path.join(OUT_PATH, "pred_gar_{:04d}.ply".format(i)))
#
#         visualize_garment_body(
#             pred_gar, body, os.path.join(OUT_PATH, "img_{:04d}.png".format(i)), garment_class='t-shirt', side='front')
#         i += 1

    # Concate frames of sequence data using this command
    # ffmpeg -r 10 -i img_%04d.png -vcodec libx264 -crf 10  -pix_fmt yuv420p check.mp4
    # Make GIF
    # convert -delay 200 -loop 0 -dispose 2 *.png check.gif
    # convert check.gif -resize 512x512 check_small.gif





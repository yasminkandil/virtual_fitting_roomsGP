import argparse
import base64
import io
import json
from flask import Flask, jsonify, request, send_file
import torch
import torch.nn as nn
from train import network
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import cv2
import uuid

import os
from train.utils import *
#@app.route('/createTex', methods=['POST'])
app = Flask(__name__)

class Demo():
    data_dir='D:/Series/shaghallll/pix2surf/test_data/images/'

 
    def __init__(self):
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
       #(ResNet) is a deep learning model used for computer vision applications.
       #It has 2 input and output channels, 2 intermediate layers, 0 padding, and uses 64 filters per layer.
        self.net_map = network.ResnetGenerator(2, 2, 0, 64, n_blocks=6, norm_layer = nn.InstanceNorm2d)
       #u-net It's one of the earlier deep learning segmentation models
        self.net_seg = network.UnetGenerator(input_nc=3, output_nc=2, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d)
    
        
    def get_args(self,id):
        ap = argparse.ArgumentParser()

        ap.add_argument("--gpus", type = str, default = '0', help = "-1 for cpu")
        ap.add_argument("--pose_id", type = str, default = id)
        ap.add_argument("--img_id", type = str, default = id)
        # ap.add_argument("--low_type", type = str, default = 'shorts', help = 'pants|shorts')

        ap.add_argument("--output", type = str, default = 'D:/Projects/fitmoi_mob_app/assets/', help = "Location where renderings are stored")
        # ap.add_argument("--video", type = str, default = './video', help = "location where text maps and videos are stored")

        ap.add_argument("--body_tex", type = str, default = 'D:/Series/shaghallll/pix2surf/test_data/images/body_tex/body_tex.jpg')

        self.opt = ap.parse_args()
   
    def get_path(self,clothtype):
        self.opt.low_mesh = 'D:/Series/shaghallll/pix2surf/test_data/meshes/' + '/lower_{}.obj'.format(self.opt.pose_id)
        self.opt.up_mesh = 'D:/Series/shaghallll/pix2surf/test_data/meshes/'  + '/upper_{}.obj'.format(self.opt.pose_id)
        self.opt.body_mesh = 'D:/Series/shaghallll/pix2surf/test_data/meshes/'  + '/body_{}.obj'.format(self.opt.pose_id)

        self.opt.img_up_front = self.data_dir   + '/'+clothtype+'{}.jpg'.format(self.opt.img_id)
        self.opt.img_up_back =  self.data_dir  + '/'+clothtype+'{}_b.jpg'.format(self.opt.img_id)

        
        self.opt.seg_up_front ='D:/Series/shaghallll/pix2surf/pretrained/seg_shirts_front.pt'
        self.opt.seg_up_back ='D:/Series/shaghallll/pix2surf/pretrained/seg_shirts_back.pt'

        self.opt.map_up_front ='D:/Series/shaghallll/pix2surf/pretrained/map_shirts_front.pt'
        self.opt.map_up_back ='D:/Series/shaghallll/pix2surf/pretrained/map_shirts_back.pt'

    
    def get_gpus(self):
        """Add device on which the code will run"""
        gpus = []
        for s in list(self.opt.gpus):
            if (s.isdigit()):
                gpus.append(int(s))
        # if gpus[0] == -1:
        self.device = torch.device("cpu")
        # else:
        #     self.device = vice("cuda", index=gpus[0])

        self.opt.gpu_ids = gpus

    def read_images(self, image_path):
        image = self.transform(Image.open(image_path).convert("RGB"))
        image = image.unsqueeze(0)
        return image.to(self.device)

    def get_img_rep(self, seg_out):
        m = torch.nn.Softmax2d()
        out = m(seg_out)
        out = out.squeeze(0)[1, :, :]
        out_fg_binary = binarizeimage(out)

        x = torch.from_numpy(np.linspace(-1, 1, 256))
        y = torch.from_numpy(np.linspace(-1, 1, 256))

        xx = x.view(-1, 1).repeat(1, 256)
        yy = y.repeat(256, 1)
        meshed = torch.cat([yy.unsqueeze_(2), xx.unsqueeze_(2)], 2)
        meshed = meshed.permute(2, 0, 1)

        out_fg_binary = out_fg_binary.unsqueeze(0)
        mask2 = torch.cat((out_fg_binary, out_fg_binary), dim=0)

        rend_rep = mask2.float() * meshed.float()
        return rend_rep.unsqueeze(0).to(self.device)


    def forward(self):
     dict = ['up_front', 'up_back']
     for val in dict:
            map_net_pth = getattr(self.opt, 'map_'+ val)
            self.net_map.load_state_dict(torch.load(map_net_pth, map_location=self.device))

            seg_net_pth = getattr(self.opt, 'seg_'+val)
            self.net_seg.load_state_dict(torch.load(seg_net_pth, map_location=self.device))

            self.net_seg.to(self.device)
            self.net_seg.eval()

            self.net_map.to(self.device)
            self.net_map.eval()

            img_path = getattr(self.opt, 'img_'+val)
            image = self.read_images(img_path)

            output = self.net_seg(image)
            map_in = self.get_img_rep(output)
            map_in = map_in.to(self.device)

            out = self.net_map(map_in)
            out = out.permute(0, 2, 3, 1)
            uv_out = F.grid_sample(image, out)
            setattr(self, 'uv_'+ val, tensor2image(uv_out[0, :, :, :]))

   
    def combine_textures(self):
        dirs = ['up']
    
        for val in dirs:
            cut1 = getattr(self, 'uv_'+val + '_front')
            cut2 = getattr(self, 'uv_' + val + '_back')
            base = np.zeros((2000, 2000, 3))
            cut1 = cv2.resize(cut1, (1000, 1000))
            cut2 = cv2.resize(cut2, (1000, 1000))
            base = base.astype('float64')
            base[500:1500, 0:1000] = cut1
            base[500:1500, 1000:2000] = cut2
            prodId = str(request.json['prodId'])

            # dynamic_id = str(uuid.uuid4())[:1]  # generate a unique id
            # save_file = os.path.join(self.opt.output, val + '_' + dynamic_id + '.jpg')
            save_file = os.path.join(self.opt.output, val + '_' + prodId + '.jpg')

            # save_file2=os.path.join(self.opt.output, val + '_' + dynamic_id + '.jpg')
   
            # save_file = os.path.join(self.opt.output, val +'.jpg')
            setattr(self.opt, 'tex_loc_' +val, save_file)
            cv2.imwrite(save_file, base)
            setattr(self, 'tex_'+val, base)
            # return dynamic_id
            return prodId

        # image = Image.open(save_file)
        # return image


        
    

    def run(self):
        self.forward()
        self.combine_textures()

@app.route('/tryy', methods=['POST'])


# def tryy():
#         data_dir='D:/Series/shaghallll/pix2surf/test_data/images/'

#         if not request.json:
#             return "not found", 400

#         id = request.json['id']
#         frontPath = request.json['frontPath']
#         backPath = request.json['backPath']
#         clothType = request.json['clothType']

#         f_imgdata = base64.b64decode(frontPath)
#         b_imgdata = base64.b64decode(backPath)

#         frontImageName = clothType + str(id) + ".jpg"
#         backImageName = clothType + str(id) + "_b.jpg"

#         with open(data_dir + frontImageName, 'wb') as f:
#             f.write(f_imgdata)
#         with open(data_dir + backImageName, 'wb') as f2:
#             f2.write(b_imgdata)

#         demo = Demo()
#         demo.get_args(id)
#         demo.get_path(clothType)
#         demo.get_gpus()
#         demo.run()
        
        
#         if not os.path.isdir(demo.opt.output):
#             os.makedirs(demo.opt.output)
            
#         # uv_front, uv_back = demo.forward()
#         print(demo.opt.output)

#         imagee ="D:/Projects/fitmoi_mob_app/assets/up"+'_' + demo.combine_textures() +".jpg"
        
#         with open(imagee, "rb") as image:
#                     f = image.read()
#                     b = bytearray(f)
#         b64Image = base64.b64encode(b)

#     # Get image data
#         #image_data = image.tobytes()

#         return jsonify({"base":b64Image.decode('utf-8'), 'Status': "Done"})
@app.route('/tryy', methods=['POST'])
def tryy():
    data_dir='D:/Series/shaghallll/pix2surf/test_data/images/'

    if not request.json:
        return "not found", 400

    id = request.json['id']
    frontPath = request.json['frontPath']
    backPath = request.json['backPath']
    clothType = request.json['clothType']
    prodId = request.json['prodId']

    # productId = request.json['productId']  # Add this line

    f_imgdata = base64.b64decode(frontPath)
    b_imgdata = base64.b64decode(backPath)

    frontImageName = clothType + str(prodId) + ".jpg"
    backImageName = clothType + str(prodId) + "_b.jpg"

    with open(data_dir + frontImageName, 'wb') as f:
        f.write(f_imgdata)
    with open(data_dir + backImageName, 'wb') as f2:
        f2.write(b_imgdata)

    demo = Demo()
    demo.get_args(prodId)
    demo.get_path(clothType)
    demo.get_gpus()
    demo.run()

    if not os.path.isdir(demo.opt.output):
        os.makedirs(demo.opt.output)

    # uv_front, uv_back = demo.forward()
    print(demo.opt.output)

    imagee = "D:/Projects/fitmoi_mob_app/assets/up" + '_' + demo.combine_textures() + ".jpg"

    with open(imagee, "rb") as image:
        f = image.read()
        b = bytearray(f)
    b64Image = base64.b64encode(b)

    # Get image data
    # image_data = image.tobytes()

    return jsonify({"base": b64Image.decode('utf-8'), 'Status': "Done", 'prodId': prodId})  # Modify this line


if __name__ == '__main__':
    demo = Demo()
    app.run(host='0.0.0.0', port=8050,debug=True)
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import random
import time
import os
import json
import sys
import torch
import re
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)
sys.path.append("E:/cs (Senior) Semester 1/gp/fashion_compatibility_mcn-master/mcn/")
import torchvision.transforms as transforms
from model import CompatModel
from utils import prepare_dataloaders
from PIL import Image

data_root = "E:/cs (Senior) Semester 1/gp/fashion_compatibility_mcn-master/data/"
img_root = os.path.join(data_root, "images\images")

#to prepare data loders for testing
_, _, _, _, test_dataset, _ = prepare_dataloaders(root_dir=img_root, num_workers=1)
#setting the device to cpu 
device = torch.device('cpu')
#this method used to move the model to cpu in case
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=2757).to(device)
# Load pretrained weights
model.load_state_dict(torch.load("E:/cs (Senior) Semester 1/gp/fashion_compatibility_mcn-master/mcn/model_train_relation_vse_type_cond_scales.pth", map_location="cpu"))
#putting the model on evaluation mode 
model.eval()
#iterating on over the named parameter in the model
for name, param in model.named_parameters():
    #checking if the named parameter does not contain fc(fully connected layers)
    if 'fc' not in name:
        #making this parameter not updatable in the training 
        param.requires_grad = False

#this function do a forward pass and backward pass capturing the intermediate tensor during backward pass
#this function takes an image tensor ,model and optional flag
def defect_detect(img, model, normalize=True):
    # Register hook for comparison matrix
    relation = None
    #this function is a hook to capture intermediate tensor during backward pass
    def func_r(module, grad_in, grad_out):
        nonlocal relation
        relation = grad_in[1].detach()

    for name, module in model.named_modules():
        #checking the named moduel if it matches it registers the func_r hook
        if name == 'predictor.0':
            module.register_backward_hook(func_r)
    #computing the score using input image tensor (this is the forward pass)
    out  = model._compute_score(img)
    out = out[0]

    #preparing one hot tensor with a negative value and moves it to cpu
    #it rests the gradient of the model
    one_hot = torch.FloatTensor([[-1]]).to(device)
    model.zero_grad()
    #performing backward pass using back probagation
    out.backward(gradient=one_hot, retain_graph=True)

    #if normalis is true it normalizes the relation by deviding it by the range of its value (min ,max)
    if normalize:
        relation = relation / (relation.max() - relation.min())
    #adding a small constant value
    relation += 1e-3
    return relation, out.item()



#taking the compatability matrix and list onf indices
def item_diagnosis(relation, select):
    
    #converting the comptability matrix into a list of matrices 
    #each matric correspond to the comptability between a pair of items in the outfit 
    mats = vec2mat(relation, select)
    for m in mats:
        
        #exludeing the diagonal elements(comptability of an item with itself)
        mask = torch.eye(*m.shape).bool()
        #the masked values are set to 0
        m.masked_fill_(mask, 0)
        
        
    #concatinating the modified matrices along the first dimension 
    #used to obtain a single vector representing the dignosis value of each item 
    #higher values indicates higher incomptability
    result = torch.cat(mats).sum(dim=0)
    
    #sorting indices of result in desnding order ,this represent items
    #from most incompatiable to least incompatable
    order = [i for i, j in sorted(enumerate(result), key=lambda x:x[1], reverse=True)]
    return result, order

#converting vector relation of length 60 into a list of 4 matrices each correspond to a layer in backend cnn
def vec2mat(relation, select): 
    mats = []
    for idx in range(4):
        #creating a 5*5 matrix of zeros
        mat = torch.zeros(5, 5)
        #filling the upper triangle of the matrix with the values of the relation=15 value
        mat[np.triu_indices(5)] = relation[15*idx:15*(idx+1)]
        #adding the transpose of the upper triangle to the lower triangle to ensure the matrix is symetric
        mat += torch.triu(mat, 1).transpose(0, 1)
        #selecting the rows and the columns of the matrix corresponding to the items spesived in select
        mat = mat[select, :]
        mat = mat[:, select]
        #appending the modified matrix
        mats.append(mat)
    return mats
#this function is resonsible for retreving the dataset to substitute the worst item for the best choice
def retrieve_sub(x, select, order, try_most=5):
    #maping item indices to their corosponding name in the outfit
    all_names = {0:'upper', 1:'bottom', 2:'shoe', 3:'bag', 4:'accessory'}
   
    best_score = -1
    best_img_path = dict()

    for o in order:
        if best_score > 0.9:
            break
        #the current problem item index
        problem_part_idx = select[o]
        #retreving all names dictionary 
        problem_part = all_names[problem_part_idx]
        #randomly samples try_most outfits from the test_dataset to find a subestitution of the current problem
        for outfit in random.sample(test_dataset.data, try_most):
            #cheking if the outfit is already compatiable
            if best_score > 0.9:
                break
            if problem_part in outfit[1]:
                #openening the problem image an convertin it to RGB
                img_path = os.path.join(test_dataset.root_dir, outfit[0], str(outfit[1][problem_part]['index'])) + '.jpg'
                img = Image.open(img_path).convert('RGB')
                #tansforming the image 
                img = test_dataset.transform(img).to(device)
                #img is assigned to input tensor x this replaces the original item with the transformed img
                x[0][problem_part_idx] = img
                with torch.no_grad():
                    #using the model to compute compatability score for the modified outfit 
                    out = model._compute_score(x)
                    score = out[0]
                #checking if the item score is better than the best score
                if score.item() > best_score:
                    best_score = score.item()
                    best_img_path[problem_part] = img_path
        #if the problem item exist in best img path it means subestitution was found with a higher score
        if problem_part in best_img_path:
            x[0][problem_part_idx] = test_dataset.transform(Image.open(best_img_path[problem_part]).convert('RGB')).to(device)
    
            print('problem_part: {}'.format(problem_part))
            print('best substitution: {} {}'.format(problem_part, best_img_path[problem_part]))
            print('After substitution the score is {:.4f}'.format(best_score))
    return best_score, best_img_path

#this function take image as bytes and encode to bease64 format 
def base64_to_tensor(image_bytes_dict):
    my_transforms = transforms.Compose([
        #resizeing the image to (224,224)
        transforms.Resize((224, 224)),
        #converting image to tensor
        transforms.ToTensor(),
    ])
    outfit_tensor = []
    for k, v in image_bytes_dict.items():
        img = base64_to_image(v)
        tensor = my_transforms(img)
        outfit_tensor.append(tensor.squeeze())
    outfit_tensor = torch.stack(outfit_tensor)
    outfit_tensor = outfit_tensor.to(device)
    return outfit_tensor

def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data).convert("RGB")
    return img
@app.route('/outfit_options', methods=['GET'])
def get_outfit_options():
    # Define file paths
    data_root = "E:/cs (Senior) Semester 1/gp/fashion_compatibility_mcn-master/data/"
    img_root = os.path.join(data_root, "images\images")

    # Load JSON data
    json_file = os.path.join(data_root, "test_no_dup_with_category_3more_name.json")
    json_data = json.load(open(json_file))
    json_data = {k: v for k, v in json_data.items() if os.path.exists(os.path.join(img_root, k))}

    #for each item outfit item it creates a dictionary entry with the label (id) and file path of the corresponding image
    #it organizes the items into their respective list
    top_options, bottom_options, shoe_options, bag_options, accessory_options = [], [], [], [], []
    for cnt, (iid, outfit) in enumerate(json_data.items()):
        if cnt > 10:
            break
        if "upper" in outfit:
            label = os.path.join(iid, str(outfit['upper']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            top_options.append({'label': label, 'value': value})
        if "bottom" in outfit:
            label = os.path.join(iid, str(outfit['bottom']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            bottom_options.append({'label': label, 'value': value})
        if "shoe" in outfit:
            label = os.path.join(iid, str(outfit['shoe']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            shoe_options.append({'label': label, 'value': value})
        if "bag" in outfit:
            label = os.path.join(iid, str(outfit['bag']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            bag_options.append({'label': label, 'value': value})
        if "accessory" in outfit:
            label = os.path.join(iid, str(outfit['accessory']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            accessory_options.append({'label': label, 'value': value})

    # Create response
    response = {
        'top_options': top_options,
        'bottom_options': bottom_options,
        'shoe_options': shoe_options,
        'bag_options': bag_options,
        'accessory_options': accessory_options
    }

    # Return response
    return jsonify(response)

@app.route('/update_top', methods=['POST'])
def update_top():
    fname = request.form.get('fname')
    content = request.form.get('content')
    name = request.form.get('name')
    date = request.form.get('date')
    
    if content is not None:
        content_type, content_string = content.split(',')
        encoded_img = base64.b64encode(content_string)
        return jsonify({'src': 'data:image/png;base64,{}'.format(encoded_img.decode())})
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return jsonify({'src': 'data:image/png;base64,{}'.format(encoded_img.decode())})
    else:
        return jsonify({'src': None})
    
@app.route('/update_bottom', methods=['POST'])
def update_bottom():
    fname = request.form.get('fname')
    content = request.form.get('content')
    name = request.form.get('name')
    date = request.form.get('date')
    
    if content is not None:
        content_type, content_string = content.split(',')
        encoded_img = base64.b64encode(content_string)
        return jsonify({'src': 'data:image/png;base64,{}'.format(encoded_img.decode())})
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return jsonify({'src': 'data:image/png;base64,{}'.format(encoded_img.decode())})
    else:
        return jsonify({'src': None})
    
@app.route('/update_shoe', methods=['POST'])
def update_shoe():
    fname = request.form.get('fname')
    content = request.form.get('content')
    name = request.form.get('name')
    date = request.form.get('date')
    
    if content is not None:
        content_type, content_string = content.split(',')
        encoded_img = base64.b64encode(content_string)
        return jsonify({'src': 'data:image/png;base64,{}'.format(encoded_img.decode())})
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return jsonify({'src': 'data:image/png;base64,{}'.format(encoded_img.decode())})
    else:
        return jsonify({'src': None})
    
@app.route('/update_bag', methods=['POST'])
def update_bag():
    fname = request.form.get('fname')
    content = request.form.get('content')
    name = request.form.get('name')
    date = request.form.get('date')
    
    if content is not None:
        content_type, content_string = content.split(',')
        encoded_img = base64.b64encode(content_string)
        return jsonify({'src': 'data:image/png;base64,{}'.format(encoded_img.decode())})
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return jsonify({'src': 'data:image/png;base64,{}'.format(encoded_img.decode())})
    else:
        return jsonify({'src': None})
@app.route('/update_accessory', methods=['POST'])
def update_accessory():
    fname = request.form.get('filename')
    content = request.form.get('contents')
    name = request.form.get('name')
    date = request.form.get('date')
    triggered = request.form.get('triggered')

    if 'upload' in triggered and content is not None:
        content_type, content_string = content.split(',')
        response = {
            'src': 'data:image/png;base64,{}'.format(content_string)
        }
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, 'rb').read())
        response = {
            'src': 'data:image/png;base64,{}'.format(encoded_img.decode())
        }
    else:
        response = {
            'src': None
        }

    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try_most = data['try_most']
    top_img = base64.b64decode(data['top_img'].split(',')[1])
    bottom_img = base64.b64decode(data['bottom_img'].split(',')[1])
    shoe_img = base64.b64decode(data['shoe_img'].split(',')[1])
    bag_img = base64.b64decode(data['bag_img'].split(',')[1])
    accessory_img = base64.b64decode(data['accessory_img'].split(',')[1])

    img_dict = {
        "top": top_img,
        "bottom": bottom_img,
        "shoe": shoe_img,
        "bag": bag_img,
        "accessory": accessory_img
    }

    img_tensor = base64_to_tensor(img_dict)
    img_tensor.unsqueeze_(0)
    relation, score = defect_detect(img_tensor, model)

    if score > 0.9:
        return jsonify({
            'message': 'This outfit is compatible.',
            'score': score
        })

    relation = relation.squeeze()
    result, order = item_diagnosis(relation, select=[0, 1, 2, 3, 4])
    best_score, best_img_path = retrieve_sub(img_tensor, [0, 1, 2, 3, 4], order, try_most)

    img_outputs = {}

    for part in ["top", "bottom", "shoe", "bag", "accessory"]:
        if part in best_img_path.keys():
            img = Image.open(best_img_path[part])
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_outputs[part] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        else:
            img_outputs[part] = data[part + '_img']

    return jsonify({
        'message': 'Revised outfit generated successfully.',
        'original_score': score,
        'revised_score': best_score,
        'img_outputs': img_outputs
    })






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050,debug=True)
    





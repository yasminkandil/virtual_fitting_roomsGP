import base64
import json
import shutil
import cv2

#from detect_body_parts.extract.demo.test import getMeasurements
from human_modelling.src.demo import show_app
#from TailorNet_master.run_tailornet import run_tailornet
#from clothes_modelling.segment.run import segmentCloth
# from clothes_modelling.create_texture.createTex import createTexture
# from flask import Flask, request, jsonify,json, make_response
# from werkzeug.serving import WSGIRequestHandler
import threading,time
import os
# WSGIRequestHandler.protocol_version = "HTTP/1.1"
# app = Flask(__name__)
# app.config['FLASK_ENV'] = 'development'
import gc
# @app.route('/get_measurement', methods=['POST'])
def get_measurement():
    # if not request.json:
    #     return "not found", 400
    # else:
    #     uniqueId = request.json['uniqueId']
    #     frontImagePath = "front_" + uniqueId + ".jpg"
    #     sideImagePath = "side_" + uniqueId + ".jpg"
    #     if 'waiting' in request.json:
    #         fileName ='data_'+str(uniqueId)+'.txt'
    #         os.listdir()
    #         if os.path.exists(fileName):
    #             with open(fileName) as json_file:
    #                 data = json.load(json_file)
    #             os.remove(frontImagePath)
    #             os.remove(sideImagePath)
    #             os.remove(fileName)
    #             return jsonify({'measurements': data, 'Status': 'Done'})
    #         else:
    #             return jsonify({'Status': 'Processing'})
    #     elif 'frontImage' in request.json:
    #         height = request.json['height']
    #         frontimageJson = request.json['frontImage']
    #         frontimgdata = base64.b64decode(frontimageJson)
    #         sideimageJson = request.json['sideImage']
    #         sideimgdata = base64.b64decode(sideimageJson)
    #         with open(frontImagePath, 'wb') as f:
    #             f.write(frontimgdata)
    #         with open(sideImagePath, 'wb') as f:
    #             f.write(sideimgdata)
            thread=myThread2(11,180, "test/front.jpg", "test/side.jpg")
            thread.start()
    #         del frontimageJson
    #         del frontimgdata
    #         del frontImagePath
    #         del sideimageJson
    #         del sideimgdata
    #         del sideImagePath
    #         gc.collect()
    #         return jsonify({'measurements':'wait'})

# @app.route('/create-model', methods=['POST'])
def create_model():
    # if not request.json :
    #     return "not found", 400
    # elif request.json and 'waited' in request.json:
    #     uniqueId = request.json['uniqueId']
    #     if os.path.exists(uniqueId+'.txt'):
    #         return jsonify({'completed':'True'})
    #     else:
    #         return jsonify({'completed': 'False'})
    # elif request.json and 'measurement' in request.json:
    #     measurements = request.json['measurement']
    #     uniqueId=request.json['uniqueId']
        thread = myThread({"height":180,"weight":80,"hip": 131.81226650059605, "chest": 113.96992934621463},"11")
        thread.start()
        # return jsonify({'processing': 'True'})

def convertb2r(bytearray):
    base64String = base64.b64encode(bytearray)
    string = base64String.decode('utf-8')
    return string
# @app.route('/fit-model', methods=['POST'])
def fit_model():
#     if not request.json :
#         return "not found", 400
#     elif request.json and 'waiting' in request.json:
#     uniqueId = 10
#         garmentClass = request.json['garmentClass']
#
#         if os.path.exists("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj") and os.path.exists("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj"):
#             with open("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj", "r") as hobject:
#                 hf = hobject.read()
#                 # hb = bytearray(hf)
#             # humanModel64 = base64.b64encode(hb)
#             with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj", "r") as gobject:
#                 gf = gobject.read()
#                 # gb = bytearray(gf)
#             # garmentModel64 = base64.b64encode(gb)
#
#             with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".mtl", "r") as GMtlobject:
#                 GMtlf = GMtlobject.read()
#             # Mtl64 = base64.b64encode(Mtlb)
#             with open("meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".mtl", "r") as HMtlobject:
#                 HMtlf = HMtlobject.read()
#
#             with open("meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".jpg", "rb") as HIMGobject:
#                 HIMGlf = HIMGobject.read()
#                 HII = bytearray(HIMGlf)
#
#             with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".jpg", "rb") as GIMGobject:
#                 GIMGlf = GIMGobject.read()
#                 GII = bytearray(GIMGlf)
#
#             # Mtl64 = base64.b64encode(Mtlb)
#             return jsonify({'completed': 'True','humanModel':hf,'garmentModel':gf,'human-mtl':HMtlf,'garment-mtl':GMtlf,'human-image':convertb2r(HII),'garment-image':convertb2r(GII)})
#         else:
#             return jsonify({'completed': 'False'})
#     elif request.json and 'uniqueId' in request.json:
    uniqueId="10"
    garmentClass="short-pant"
#         textureId=request.json['textureId']
#         if os.path.exists("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj") and os.path.exists("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj"):
#
#             with open("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj", "r") as hobject:
#                 hf = hobject.read()
#                 # hb = bytearray(hf)
#             # humanModel64 = base64.b64encode(hb)
#             with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj", "r") as gobject:
#                 gf = gobject.read()
#                 # gb = bytearray(gf)
#             # garmentModel64 = base64.b64encode(gb)
#
#             with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".mtl", "r") as GMtlobject:
#                 GMtlf = GMtlobject.read()
#             # Mtl64 = base64.b64encode(Mtlb)
#             with open("meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".mtl", "r") as HMtlobject:
#                 HMtlf = HMtlobject.read()
#             os.remove("meshes/" + uniqueId + "/" + garmentClass + "_" + uniqueId + ".jpg")
#             shutil.copyfile("Textures/texture_" + textureId + ".jpg",
#                             "meshes/" + uniqueId + "/" + garmentClass + "_" + uniqueId + ".jpg")
#             with open("meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".jpg", "rb") as HIMGobject:
#                 HIMGlf = HIMGobject.read()
#                 HII = bytearray(HIMGlf)
#             with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".jpg", "rb") as GIMGobject:
#                 GIMGlf = GIMGobject.read()
#                 GII = bytearray(GIMGlf)
#             # Mtl64 = base64.b64encode(Mtlb)
#             return jsonify({'completed': 'True','humanModel':hf,'garmentModel':gf,'human-mtl':HMtlf,'garment-mtl':GMtlf,'human-image':convertb2r(HII),'garment-image':convertb2r(GII)})
#
    thread = myThread3(uniqueId,garmentClass)
    thread.start()
#         return jsonify({'processing': 'True'})
#
#
# @app.route('/change-skin', methods=['POST'])
# def change_skin():
#     if not request.json:
#         return "not found", 400
#     else:
#         skinNumber = request.json['skinNumber']
#         uniqueId = request.json['uniqueId']
#         os.remove("meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".jpg")
#         shutil.copyfile("skins/skin_"+skinNumber+".jpg", "meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".jpg")
#         with open("meshes/" + uniqueId + "/" + "body" + "_" + uniqueId + ".jpg", "rb") as HIMGobject:
#             HIMGlf = HIMGobject.read()
#             HII = bytearray(HIMGlf)
#         return jsonify({'completed': 'True','human-image':convertb2r(HII)})
#
# @app.route('/get_uv_tex_map', methods=['POST'])
# def get_uv_tex_map():
#     if not request.json:
#         return "not found", 400
#     else:
#         data_dir='Textures/'
#         id = request.json['uid']
#         if 'waiting' in request.json:
#             imageName=str(id) + ".jpg"
#             outputImagePath = data_dir+"texture_" + imageName
#             if os.path.exists(outputImagePath):
#                 with open(outputImagePath, "rb") as image:
#                     f = image.read()
#                     b = bytearray(f)
#                 b64Image = base64.b64encode(b)
#                 os.remove(data_dir+'front_'+imageName)
#                 os.remove(data_dir+'back_'+imageName)
#                 return jsonify({'texture': b64Image.decode('utf-8'), 'Status': 'Done'})
#             else:
#                 return jsonify({'Status': 'Processing'})
#         elif 'frontImage' in request.json:
#             clothType = request.json['clothType']
#             id = request.json['uid']
#             f_image = request.json['frontImage']
#             b_image = request.json['backImage']
#             f_imgdata = base64.b64decode(f_image)
#             b_imgdata = base64.b64decode(b_image)
#             # id=1
#             frontImageName = 'front_' + str(id) + ".jpg"
#             backImageName = 'back_' + str(id) + ".jpg"
#             with open(data_dir+frontImageName, 'wb') as f:
#                 f.write(f_imgdata)
#             with open(data_dir+backImageName, 'wb') as f2:
#                 f2.write(b_imgdata)
#             Thread=myThread4(id,frontImageName,backImageName,clothType)
#             Thread.start()
#             del f_image
#             del b_image
#             del f_imgdata
#             del b_imgdata
#             del frontImageName
#             del backImageName
#             with open("clothes_modelling/"+clothType+'Obj.obj', "rb") as object:
#                 f_object = object.read()
#                 b_object = bytearray(f_object)
#             b64OObject = base64.b64encode(b_object)
#             with open("clothes_modelling/"+clothType+'Mat.mtl', "rb") as mtl:
#                 f_mtl = mtl.read()
#                 b_mtl = bytearray(f_mtl)
#             b64Omtl = base64.b64encode(b_mtl)
#             gc.collect()
#             return jsonify({'Status': 'processing','object':b64OObject.decode('utf-8'),'mtl':b64Omtl.decode('utf-8')})

class myThread (threading.Thread):
   def __init__(self,measurements,uniqueId):
      threading.Thread.__init__(self)
      self._measurements=measurements
      self._uniqueId=uniqueId
   def run(self):
       show_app(self._measurements,self._uniqueId)
       del self
       gc.collect()
class myThread3 (threading.Thread):
   def __init__(self,uniqueId,garmentClass):
      threading.Thread.__init__(self)
      self._uniqueId=uniqueId
      self._garmentClass=garmentClass
   def run(self):
       run_tailornet(self._uniqueId,self._garmentClass,)
       del self
       gc.collect()

class myThread2 (threading.Thread):
   def __init__(self,id,height,frontImage,sideImage):
      threading.Thread.__init__(self)
      self._id = id
      self._frontImage = frontImage
      self._sideImage = sideImage
      self._height = height
   def run(self):
       getMeasurements(self._id,self._height,self._frontImage,self._sideImage)
       print("finish")

class myThread4 (threading.Thread):
   def __init__(self,id,frontImageName,backImageName,clothType):
      threading.Thread.__init__(self)
      self._id = id
      self._frontImageName = frontImageName
      self._backImageName = backImageName
      self._clothType = clothType
   def run(self):
       data_dir ="Textures/"
       segmentCloth(self._frontImageName, id)
       segmentedFront = cv2.imread(data_dir+"seg_" + self._frontImageName, 0)
       segmentCloth(self._backImageName, id)
       segmentedBack = cv2.imread(data_dir+"seg_" + self._backImageName, 0)
       createTexture(self._clothType, data_dir+self._frontImageName, data_dir+self._backImageName, segmentedFront, segmentedBack, self._id)
       # renderObject("texture_" + str(id) + ".jpg",'create_texture/object/'+str(self._clothType)+'.obj',self._id,self._clothType)
       os.remove(data_dir+"seg_" + self._frontImageName)
       os.remove(data_dir+"seg_" + self._backImageName)

#get_measurement()
create_model()
#fit_model()
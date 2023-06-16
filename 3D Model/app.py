import base64
import json
import shutil
import cv2

from detect_body_parts.extract.demo.test import getMeasurements
from human_modelling.src.demo import show_app
from TailorNet_master.run_tailornet import run_tailornet
from flask import Flask, request, jsonify,json, make_response
# from werkzeug.serving import WSGIRequestHandler
import threading,time
import os


from human_modelling.src.demo import show_app
# WSGIRequestHandler.protocol_version = "HTTP/1.1"
app = Flask(__name__)
# app.config['FLASK_ENV'] = 'development'
import gc
@app.route('/get_measurement', methods=['POST'])
def get_measurement():
    if not request.json:
        return "not found", 400
    else:
        uniqueId = request.json['uniqueId']
        frontImagePath = "front_" + uniqueId + ".jpg"
        sideImagePath = "side_" + uniqueId + ".jpg"
        backImagePath = "back_" + uniqueId + ".jpg"

        if 'waiting' in request.json:
            fileName = 'data_' + str(uniqueId) + '.txt'
            if os.path.exists(fileName):
                with open(fileName) as json_file:
                    data = json.load(json_file)
                return jsonify({'measurements': data, 'Status': 'Done'})
            else:
                return jsonify({'Status': 'Processing'})
        elif 'frontImage' in request.json:
            height = request.json['height']

            frontimageJson = request.json['frontImage']
            frontimgdata = base64.b64decode(frontimageJson)
            sideimageJson = request.json['sideImage']
            sideimgdata = base64.b64decode(sideimageJson)
            backimageJson = request.json['backImage']
            backimgdata = base64.b64decode(backimageJson)
            with open(frontImagePath, 'wb') as f:
                f.write(frontimgdata)
            with open(sideImagePath, 'wb') as f:
                f.write(sideimgdata)
            with open(backImagePath, 'wb') as f:
                f.write(backimgdata)
            thread = myThread2(uniqueId, height, frontImagePath, sideImagePath, backImagePath)
            thread.start()
            del frontimageJson
            del frontimgdata
            del frontImagePath
            del sideimageJson
            del sideimgdata
            del sideImagePath
            del backimgdata
            del backImagePath
            gc.collect()

            # Check if the thread has terminated due to no human detection
            if not thread.is_alive():
                return jsonify({'Status': 'NoHuman'})

            return jsonify({'measurements': 'wait'})

@app.route('/create-model', methods=['POST'])
def create_model():
    if not request.json :
        return "not found", 400
    elif request.json and 'waited' in request.json:
        uniqueId = request.json['uniqueId']
        if os.path.exists(uniqueId+'.txt'):
            return jsonify({'completed':'True'})
        else:
            return jsonify({'completed': 'False'})
    elif request.json and 'measurement' in request.json:
        measurements = request.json['measurement']
        uniqueId=request.json['uniqueId']
        gender=request.json['gender']
        thread = myThread(measurements,uniqueId,gender)
        thread.start()
        return jsonify({'completed': 'True'})

def convertb2r(bytearray):
    base64String = base64.b64encode(bytearray)
    string = base64String.decode('utf-8')
    return string
@app.route('/fit-model', methods=['POST'])
def fit_model():
    if not request.json :
        return "not found", 400
    elif request.json and 'waiting' in request.json:
        uniqueId = request.json['uniqueId']
        garmentClass = request.json['garmentClass']
        gender=request.json['gender']
        texturee=request.json['texture']
        textureee = base64.b64decode(texturee)
        with open('E:/GP/fitmoi_mob_app/assets/'+garmentClass+"_"+uniqueId+".jpg", 'wb') as f:
            f.write(textureee)

        if os.path.exists("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj") and os.path.exists("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj") and os.path.exists("E:/GP/fitmoi_mob_app/assets/"+garmentClass+"_"+uniqueId+".obj") and os.path.exists("E:/GP/fitmoi_mob_app/assets/"+"body_"+uniqueId+".obj"):
            with open("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj", "r") and open("E:/GP/fitmoi_mob_app/assets/"+"body_"+uniqueId+".obj", "r") as hobject:
                hf = hobject.read()
               
            with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj", "r") and open("E:/GP/fitmoi_mob_app/assets/"+garmentClass+"_"+uniqueId+".obj", "r")  as gobject:
                gf = gobject.read()
            

            with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".mtl", "r") and open("E:/GP/fitmoi_mob_app/assets/"+garmentClass+"_"+uniqueId+".mtl", "r") as GMtlobject:
                GMtlf = GMtlobject.read()
           
            with open("meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".mtl", "r") and open("E:/GP/fitmoi_mob_app/assets/"+"body"+"_"+uniqueId+".mtl", "r") as HMtlobject:
                HMtlf = HMtlobject.read()

            with open("meshes/"+uniqueId+"/"+"body"+"_"+uniqueId+".jpg", "rb") and open("E:/GP/fitmoi_mob_app/assets/"+"body"+"_"+uniqueId+".jpg", "rb")  as HIMGobject:
                HIMGlf = HIMGobject.read()
                HII = bytearray(HIMGlf)

            with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".jpg", "rb") and open("E:/GP/fitmoi_mob_app/assets/"+garmentClass+"_"+uniqueId+".jpg", "rb") as GIMGobject:
                GIMGlf = GIMGobject.read()
                GII = bytearray(GIMGlf)

           
            return jsonify({'completed': 'True'})
        else:
            return jsonify({'completed': 'False'})
    elif request.json and 'uniqueId' in request.json:
        uniqueId = request.json['uniqueId']
        garmentClass = request.json['garmentClass']
        gender=request.json['gender']
        if os.path.exists("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj") and os.path.exists("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj") and os.path.exists("E:/GP/fitmoi_mob_app/assets/"+garmentClass+"_"+uniqueId+".obj") and os.path.exists("E:/GP/fitmoi_mob_app/assets/"+"body_"+uniqueId+".obj"):

            with open("meshes/"+uniqueId+"/"+"body_"+uniqueId+".obj", "r") and open("E:/GP/fitmoi_mob_app/assets/"+"body_"+uniqueId+".obj", "r") as hobject:
                hf = hobject.read()
              
            with open("meshes/"+uniqueId+"/"+garmentClass+"_"+uniqueId+".obj", "r") and open("E:/GP/fitmoi_mob_app/assets/"+garmentClass+"_"+uniqueId+".obj", "r") as gobject:
                gf = gobject.read()
            return jsonify({'completed': 'True'})

    thread = myThread3(uniqueId,garmentClass,gender)
    thread.start()
    return jsonify({'completed': 'True'})

class myThread (threading.Thread):
   def __init__(self,measurements,uniqueId,gender):
      threading.Thread.__init__(self)
      self._measurements=measurements
      self._uniqueId=uniqueId
      self._gender=gender
   def run(self):
       show_app(self._measurements,self._uniqueId,self._gender)
       del self
       gc.collect()
class myThread3 (threading.Thread):
   def __init__(self,uniqueId,garmentClass,gender):
      threading.Thread.__init__(self)
      self._uniqueId=uniqueId
      self._garmentClass=garmentClass
      self._gender=gender
   def run(self):
       run_tailornet(self._uniqueId,self._garmentClass,self._gender)
       del self
       gc.collect()

class myThread2(threading.Thread):
    def __init__(self, id, height, frontImage, sideImage, backImage):
        threading.Thread.__init__(self)
        self._id = id
        self._frontImage = frontImage
        self._backImage = backImage
        self._sideImage = sideImage
        self._height = height
        self._isHuman = True
        #self._is_running=True

    def run(self):
        is_human = getMeasurements(self._id, self._height, self._frontImage, self._sideImage, self._backImage)
        if not is_human:
            return
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,debug=True)
    
import base64
import json
import shutil
import cv2

#from detect_body_parts.extract.demo.test import getMeasurements
#from human_modelling.src.demo import show_app
from TailorNet_master.run_tailornet import run_tailornet

import threading,time
import os
import gc

def get_measurement():
    
            thread=myThread2(31,176, "test/frontF.jpg", "test/sideF.jpg","test/backF.jpg")
            thread.start()
    
def create_model():
    
        thread = myThread({"height":174,"weight":65,"hip": 73.30642014758956, "chest":  94.64917840576544},"31")
        thread.start()
       

def fit_model():

    uniqueId="31"
    garmentClass="t-shirt"

    thread = myThread3(uniqueId,garmentClass)
    thread.start()

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
   def __init__(self,id,height,frontImage,sideImage,backImage):
      threading.Thread.__init__(self)
      self._id = id
      self._frontImage = frontImage
      self._sideImage = sideImage
      self._backImage = backImage
      self._height = height
      


   def run(self):
       getMeasurements(self._id,self._height,self._frontImage,self._sideImage,self._backImage)
       print("finish")

#get_measurement()
#create_model()
fit_model()
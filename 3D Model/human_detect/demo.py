#!/usr/bin/python
# coding=utf-8
# from flask import jsonify
import numpy as np
from ..src.maya_widget import MayaviQWidget
from ..src import utils
import os
def show_app(measurements,uniqueId):
  weight=float(measurements['weight'])
  height=float(measurements['height'])
  if measurements['chest'] != None:
    chest=float(measurements['chest'])
  else:
    chest=0
  if measurements['hip'] != None:
    hip=float(measurements['hip'])
  else:
    hip=0
  if measurements['back'] != None:
    back=float(measurements['back'])
  else:
    back=0 

  
  viewer3D = MayaviQWidget()
  data = []
  data.append(weight ** (1.0 / 3.0) * 1000)
  data.append(height * 10)
  for i in range(2, 19):
    data.append(0)
  data[3]=chest*10;
  data[5]=hip*10;
  data[4]=back*10;
  data = np.array(data).reshape(utils.M_NUM, 1)
  viewer3D.select_mode(label='male')
  [t_data, value] = viewer3D.predict(data)
  shape_parameters=[]
  shape_parameters.append(t_data[1])  # height
  shape_parameters.append(t_data[0] * -1)  # weight
  shape_parameters.append(t_data[8])  # shoulder
  shape_parameters.append(t_data[16])  # lower body height
  shape_parameters.append(t_data[2])  # neck
  shape_parameters.append(t_data[10])  # waist
  shape_parameters.append(t_data[18])  # leg
  shape_parameters.append(t_data[9])  # upper body height
  shape_parameters.append(t_data[14])  # arm
  shape_parameters.append(t_data[3])  # chest
  with open("shape_"+uniqueId+'.txt', 'w') as f:
    for item in shape_parameters:
      f.write("%f " % item)
  if os.path.exists(uniqueId + '.txt'):
    os.remove(uniqueId + '.txt')

  os.rename("shape_"+uniqueId+'.txt',uniqueId+'.txt')

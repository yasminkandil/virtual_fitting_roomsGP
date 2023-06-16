import numpy as np
import os
from human_modelling.src.reshaper import Reshaper
import human_modelling.src.utils as utils
os.environ['ETS_TOOLKIT'] = 'qt4'


class MayaviQWidget():
  def __init__(self):
    self.bodies = {"female": Reshaper(label="female"), "male":Reshaper(label="male")}
    self.body = self.bodies["male"]
    self.flag_ = 0
    self.vertices = self.body.mean_vertex
    self.normals = self.body.normals
    self.facets = self.body.facets
    self.input_data = np.zeros((utils.M_NUM, 1))
    self.update()

  def update(self):
    [self.vertices, self.normals, self.facets] = \
        self.body.mapping(self.input_data, self.flag_)
    self.vertices = self.vertices.astype('float32')

  def select_mode(self, label="female", flag=0):
    self.body = self.bodies[label]
    self.flag_ = flag
    self.update()

  def save(self,uniqueId,t_data):
    f = open("../src/"+uniqueId+".obj", "w+")
    del f
    utils.save_obj("../src/"+uniqueId+".obj", self.vertices, self.facets+1)
    output = np.array(utils.calc_measure(self.body.cp, self.vertices, self.facets),dtype=np.float32)
    for i in range(0, utils.M_NUM):
      print("%s: %f: %f" % (utils.M_STR[i], output[i, 0]/1000,t_data[i]))

  def predict(self, data):
    mask = np.zeros((utils.M_NUM, 1), dtype=bool)
    for i in range(0, data.shape[0]):
      if data[i, 0] != 0:
        data[i, 0] -= self.body.mean_measure[i, 0]
        data[i, 0] /= self.body.std_measure[i, 0]
        mask[i, 0] = 1
    self.input_data = self.body.get_predict(mask, data)
    self.update()
    measure = self.body.mean_measure + self.input_data*self.body.std_measure
    return [self.input_data, measure]

__author__ = 'tr1c4011'
import numpy as np
from numpy.linalg import inv

def loss_function(param_vec, input_matrix, actual_value_vec):

   loss = np.dot(np.transpose(actual_value_vec - (np.dot(input_matrix, param_vec))), (actual_value_vec - (np.dot(input_matrix, param_vec))))
   return loss

def derivative(input_matrix, actual_values_vec):
  teta =   np.dot(inv(np.dot(np.transpose(input_matrix), input_matrix)) ,np.dot(np.transpose(input_matrix), actual_values_vec))
  return teta




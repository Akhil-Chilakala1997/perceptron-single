import numpy as np
import logging as lg
from tqdm import tqdm
class Perceptron :
    def __init__(self,eta,epoch):
        self.weights = np.random.randn(3)*1e-4
        self.eta=eta
        self.epoch=epoch    
    
    def activation(self,weights,inputs):
        z = np.dot(inputs,self.weights)#Z=W*X
        return np.where(z>0,1,0)
    def fit(self,x,y):
        self.x = x
        self.y = y
        x_with_bias = np.c_[self.x,-np.ones((len(self.x),1))]
        lg.info(f"X with bias: \n{x_with_bias}")
        for i in tqdm( range(self.epoch), total = self.epoch,desc = "training the model"):
            lg.info('*'*10)
            lg.info(f'epoch-  {i}')
            lg.info('*'*10)
            y_hat = self.activation(self.weights,x_with_bias)#forward prop
            self.error = self.y-y_hat
            lg.info(f'error-->   {self.error}')
            self.weights = self.weights+self.eta*np.dot(x_with_bias.T,self.error)#backward prop
            lg.info(f"updated weights after epoch:\n{self.epoch}/{self.epoch} : \n{self.weights}")
    
    def predict(self,x):
        x_with_bias = np.c_[x,-np.ones((len(self.x),1))]
        return self.activation(self.weights,x_with_bias)
    def total_loss(self):
        total_loss = np.sum(self.error)
        lg.info(f"total loss: {total_loss}")
        return total_loss




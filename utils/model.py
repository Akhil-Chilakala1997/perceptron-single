import numpy as np
class Perceptron :
    def __init__(self,eta,epoch):
        self.weights = np.random.randn(3)*1e-4
        self.eta=eta
        self.epoch=epoch
    
    def activation(self,weights,inputs):
        z = np.dot(self.weights,inputs)#Z=W*X
        return np.where(z>0,1,0)
    def fit(self,x,y):
        self.x = x
        self.y = y
        x_with_bias = np.c_[self.x,-np.ones((len(self.x),1))]
        print(f"X with bias: \n{x_with_bias}")
        for i in range(self.epoch):
            print('*'*10)
            print(f'epoch-  {i}')
            print('*'*10)
            y_hat = self.activation(self.weights,x_with_bias)#forward prop
            self.error = self.y-y_hat
            print(f'error-->   {self.error}')
            self.weights = self.weights+self.eta*np.dot(x_with_bias.T,self.error)#backward prop
            print(f"updated weights after epoch:\n{self.epoch}/{self.epochs} : \n{self.weights}")
    
    def predict(self,x):
        x_with_bias = np.c_[self.x,-np.ones((len(self.x),1))]
        return self.activation(self.weights,x_with_bias)
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"total loss: {total_loss}")
        return total_loss




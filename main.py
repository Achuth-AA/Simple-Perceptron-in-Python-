from pyexpat import model
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns 
import sys 
import os  
import joblib
import math
from matplotlib.colors import ListedColormap

plt.style.use("fivethirtyeight")

class Perceptron:
    def __init__(self, eta: float=None, epochs: int=None):
        self.weights = np.random.randn(3)*1e-4
        fitting  = (eta is not None) and (epochs is not None)
        if fitting:
            print(f"initial Weights are : \n{self.weights}")
        self.epochs = epochs
        self.eta = eta


    def _zValue(self,x,w):
        return np.dot(x,w)


    def acivation_function(self,x):
        return np.where(x>0,1,0)


    def fit(self,x,y):
        self.x = x
        self.y = y 
        x_with_bias = np.c_[self.x , -np.ones((len(self.x),1))]
        print(f"X with the bias value : \n{x_with_bias}")

        for epoch in range(self.epochs):
            print("-"*50)
            print(f"epoch no :- {epoch+1}")
            print("-"*50)

            #calculating the z value {dot product of weights and inputs}
            z_val = self._zValue(x_with_bias,self.weights)
            
            #applying the activation function and finding the predicted value 
            y_hat = self.acivation_function(z_val)
            print(f"predicted value after the forward propagation \n{y_hat}")

            #calculating the error value 
            self.error = self.y - y_hat
            print(f"error: \n{self.error}")

            #weights updatation 
            self.weights = self.weights + self.eta * np.dot(self.error, x_with_bias)
            print(f"update weights after the epoch : {epoch+1}/{self.epochs}  : \t {self.weights}") 
            print("#"*50)


    # predicting the values for the test data 
    def predict(self,test_x):
        x_with_bias = np.c_[test_x , -np.ones((len(test_x),1))]
        z = self._zValue(x_with_bias,self.weights)
        return self.acivation_function(z)

    # calculating the total loss value 
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"\n total_loss: {total_loss}")
        return total_loss

    # calculating the avg loss value 
    def avg_loss(self):
        avg_loss = np.sum(self.error)/len(self.x)
        print(f"\n total_loss: {avg_loss}")
        return avg_loss


    def _create_dir_return_path(self,model_dir,filename):
        os.makedirs(model_dir,exist_ok=True)
        return os.path.join(model_dir,filename)

    #saving the joblib file in a directory  in your working directory by creatig a another folder
    def save(self,filename,model_dir=None):
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir,filename)
            joblib.dump(self,model_file_path)
        else:
            model_file_path = self._create_dir_return_path("model",filename)
            joblib.dump(self,model_file_path)
    
    #loading the dataset using joblib 
    def load(self,filepath):
        return joblib.load(filepath)


# preparing the dataset

def feature_engg(df, target_col="y"):
    x = df.drop(target_col,axis=1)
    y = df[target_col]
    return x,y


# making the "and" , "or" , "xor" gate dataset 

And  = {
    "x1" : [0,0,1,1],
    "x2" : [1,0,1,0],
    "y" : [0,0,1,0]
}

Or  = {
    "x1" : [0,0,1,1],
    "x2" : [1,0,1,0],
    "y" : [1,0,1,1]
}

Xor  = {
    "x1" : [0,0,1,1],
    "x2" : [1,0,1,0],
    "y" : [1,0,0,1]
}


and_dataset = pd.DataFrame(And)
or_dataset = pd.DataFrame(Or)
xor_dataset = pd.DataFrame(Xor)

# testing the perceptron class with the and , or and xor dataset

# x,y = feature_engg(and_dataset)
# ETA = 0.1
# EPOCHS = 10
# model_ = Perceptron(eta=ETA,epochs=EPOCHS)
# model_.fit(x,y)

x,y = feature_engg(or_dataset)
ETA = 0.1
EPOCHS = 10
model_ = Perceptron(eta=ETA,epochs=EPOCHS)
model_.fit(x,y)

# x,y = feature_engg(xor_dataset)
# ETA = 0.1
# EPOCHS = 20
# model_ = Perceptron(eta=ETA,epochs=EPOCHS)
# model_.fit(x,y)

#saving the model 
# model_.save(filename="and.model")

#saving the or model 
#it is a binary file so no need to worry about the extension of the model
# model_.save(filename="ormodel" , model_dir="ormodel")


#using the saved model 
# load_and_model = Perceptron().load(filepath='ormodel/ormodel')
# print(load_and_model.predict(test_x=[[1,0]]))
# print(load_and_model.predict(test_x=[[1,1]]))
# print(load_and_model.predict(test_x=[[0,0]]))
# print(load_and_model.predict(test_x=[[0,1]]))


def save_plot(dataset, trained_model, filename="plot.png",plot_dir="Visualizations" ):
    
    def _create_baseplot(dataset):
        dataset.plot(kind="scatter",x="x1",y="x2",c="y",s= 100,cmap="coolwarm")
        plt.axhline(y=0,color="black",linestyle="--",linewidth="1")
        plt.axvline(x=0,color="black",linestyle="-",linewidth="1")

        figure = plt.gcf()
        figure.set_size_inches(10,8)

    def _plot_decision_regions(x,y,classifier,resolution=0.02):
        color = ("lightgreen","skyblue")
        cmap = ListedColormap(color)

        x = x.values
        x1 = x[:,0]
        x2 = x[:,1]

        x1_min , x1_max = x1.min()-1, x1.max()+1
        x2_min , x2_max = x2.min()-1, x2.max()+1

        xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

        y_hat = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)

        plt.contourf(xx1,xx2,y_hat,alpha=0.3,cmap=cmap)
        plt.xlim(xx1.min(),xx1.max())
        plt.ylim(xx2.min(),xx2.max())
        plt.plot()
    
    x,y = feature_engg(dataset)
    _create_baseplot(dataset)
    _plot_decision_regions(x,y,trained_model)

    os.makedirs(plot_dir, exist_ok=True )
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)


save_plot(or_dataset,model_,filename="figure.png")
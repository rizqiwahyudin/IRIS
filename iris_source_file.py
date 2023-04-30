# %% [markdown]
# # Jupyter Notebook for Iris Class
# This notebook was made for the easy execution of the Iris classification code. The following code cell represents the design of our LDC class. 

# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sn
import matplotlib.mlab as mlab
#from scipy.stats import norm
import scipy.stats
from scipy.stats import norm
import os

# %% [markdown]
# # LDC CLASS
# The following code is the LDC class. 

# %%
#The LDC is initialised by the following: 
# - feeding a training set
# - feeding then an r_k list 
# - alpha value
# - number of iterations corresponding to the alpha value
# - list of features.


#Instantiating the class is done as follows, 
#   x = LDC(train, test, t_l, iterations, alpha, list_of_features)
#   We then call a class function called train() to execute the classification: y = x.train()



class LDC:

    def __init__(self, train, test, t_k, iterations, alpha, list_of_features):
        '''
        function init: intialises the class variables. 
        param self: self-referring parameter. Essential to use access class variables and functions. 
        param train: training data.
        param test: test data.
        param t_k: list of all the true labels.
        param iterations: number of iterations.
        param alpha: set of alpha values used in the program.
        param list_of_features: list of features that are used during the LDC.


        '''
        
        
        #class attributes are listed here. The following annotated comments show additional information about the attributes. 

        self.train = train #must be a np.array
        self.test = test #must be a np.array
        self.t_k = t_k #list type has no effect. These are the true labels for the train set. 
        self.iterations = iterations #must be an int
        self.alpha = alpha #list type has no effect. These are the true labels for the train set. 
        self.list_of_features = list_of_features #list type has no effect. These are the true labels for the train set. 
        self.class_names = ['Setosa', 'Versicolor', 'Virginica'] #class names for the program 
        self.features = len(self.list_of_features) +1 #number of features, including the bias.
        self.classes = 3 #class count
        self.weights = np.zeros((self.classes,self.features)) #uses a numpy array to set all the weights to 0. Here we have 3 classes and 5 features.
        self.g_k = np.zeros(self.classes) #sets up array for discriminant values
        self.mses = np.zeros(self.iterations) #sets up array for mse values
        self.confusion_matrix = np.zeros((self.classes,self.classes)) #sets up array for confusion matrix
    
    #the usual get and set functions.
    # -----------------------------------------#   
    def set_iterations(self, iterations):
        self.iterations = iterations
    
    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def set_train(self, train):
        self.train = train

    def set_test(self, test):
        self.test = test
    
    def set_train_test(self,train,test):
        self.train = train
        self.test = test
    
    def set_tk(self, tk):
        self.t_k = tk
    
    def set_list_of_features(self, list_of_features):
        self.list_of_features = list_of_features
    
    def set_num_of_classes(self,classes):
        self.classes = classes

    def get_iterations(self):
        return self.iterations
    
    def get_alpha(self):
        return self.alpha

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

    def get_train_test(self):
        return self.train, self.test

    def get_weights(self):
        print(self.weights)
        return self.weights

    def get_tk(self):
        return self.t_k

    def get_list_of_features(self):
        return self.list_of_features

    def get_num_of_classes(self):
        return self.classes
    
     # -----------------------------------------#
    #a function to reset the confusion matrix
    def reset_cm(self):
        print('Processing confusion matrix reset to 0.')
        self.confusion_matrix = np.zeros((self.classes,self.classes))

     
    #Implementation of equation 3.20 in the compendium.
    def sigmoid_function(self, x):

        return np.array(1/(1+ np.exp(-x)))

    #Implementation of equation 3.21 in the compendium.
    def grad_gk_mse_f(self, g_k, t_k):

        grad = np.multiply((g_k-t_k),g_k)
        return grad
    

    #calculation the gradient_w z_k, part of eq:3.21 compendium
    def grad_W_zk_f(self, x):

        grad = x.reshape(1,self.features)
        return grad
    #calculation the gradient_W mse, eq:3.22 compendium
    def grad_W_MSE_f(self, g_k, grad_gk_mse, grad_W_zk):

        return np.matmul(np.multiply(grad_gk_mse,(1-g_k)),grad_W_zk)
    #calculation MSE, eq:3.19
    def MSE_f(self, g_k,t_k):

        return 0.5*np.matmul((g_k-t_k).T,(g_k-t_k))


    #training the model
    def train_model(self):
        print(f'The model is now in training with an alpha value of ={self.alpha}. and number of iterations = {self.iterations}.') 
        #setting g_k to 1, this is the bias
        self.g_k[0] = 1
        #looping through the iterations
        for i in range(self.iterations):
            #setting start values, and resetting these every iteration
            grad_W_MSE = 0
            MSE = 0
            k = 0 #target class identifier variable
            
            for j, x in enumerate(self.train): 
                if j%30==0 and j!=0:
                    k += 1
                #iterating and calculating the g_k values
                self.g_k = self.sigmoid_function(np.matmul(self.weights,x.reshape(self.features,1)))
                #adding the MSE to the total MSE
                MSE += self.MSE_f(self.g_k,self.t_k[k])
                grad_gk_mse = self.grad_gk_mse_f(self.g_k,self.t_k[k])
                grad_W_zk = self.grad_W_zk_f(x)
                grad_W_MSE += self.grad_W_MSE_f(self.g_k, grad_gk_mse, grad_W_zk)
            #adding the MSE to the array of MSEs for plotting later
            self.mses[i] = MSE[0]
            #Updating the weights after each iteration
            self.weights = self.weights-self.alpha*grad_W_MSE



            #progress marker
            if(100*i /self.iterations) % 10 == 0:  
                print(f"\rThe program is now at: {100 * i / self.iterations}%", end='\n')
        
        print(f"\rThe program has passed {(i+1)/self.iterations *100}%", end='\n')
        print('Done!')
        #returning weights to display the matrices
        return self.weights
    



    #testing the model
    def test_model(self):
        #validating that the model is trained and that the confusion matrix is reset
        if(np.all((self.weights==0 ))):
            print('You need to train the model first')
            return False
        if(np.all((self.confusion_matrix != 0))):
            print('You have to reset the confusion matrix first')
            print('Resetting confusion matrix')
            self.reset_cm()

        print(f'The model is now in testing with an alpha value of ={self.alpha}. and number of iterations = {self.iterations}.') 
        #working with confusion matrix, adding prediction and label to the matrix
        for clas, test_set in enumerate(self.test):
            for row in test_set:
                prediction = np.argmax(np.matmul(self.weights,row))
                self.confusion_matrix[clas,prediction] += 1

        return self.confusion_matrix
    #printing the confusion matrix
    def print_confusion_matrix(self):


        print(self.confusion_matrix)
        dia_sum = 0
        for i in range(len(self.confusion_matrix)):
            dia_sum += self.confusion_matrix[i, i]
        error = 1 - dia_sum / np.sum(self.confusion_matrix)
        #printing out the error rate
        print(f'error rate = {100 * error:.1f}%')

    #plotting the confusion matrix
    def plot_confusion_matrix(self, name='ok', save=False):
        dia_sum = 0
        for i in range(len(self.confusion_matrix)):
            dia_sum += self.confusion_matrix[i, i]
        error = 1 - dia_sum / np.sum(self.confusion_matrix)

        df_cm = pd.DataFrame(self.confusion_matrix, index = [i for i in self.class_names],
                  columns = [i for i in self.class_names])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap="YlOrRd")
        plt.title(f'Confusion Matrix with the following params: \n iteration: {self.iterations}, alpha: {self.alpha}.\n error rate = {100 * error:.1f}%')
        if save:
            plt.savefig(f'confusionmatrixIris_{name}_it{self.iterations}_alpha{self.alpha }.png',dpi=200)
        else:
            plt.show()
        plt.clf()
        plt.close()
    
    #plotting the MSE
    def plot_MSE(self, save=False, log=False):
        plt.plot(self.mses)
        plt.title(f'MSE\n iteration: {self.iterations}, alpha: {self.alpha}.')
        plt.xlabel('Iteration number')
        plt.ylabel('Mean square error (MSE)')
        plt.grid('on')
        if log:
            plt.xscale('log')
        if save:
            plt.savefig(f'mse_it{self.iterations}_alpha{self.alpha}.png',dpi=200)
        else:
            plt.show()

# %% [markdown]
# # PLOTTING FUNCTIONS FOR THE LDC
# 

# %%

#plotting MSE for the alphas chosen 
def plot_mses_array(arr, alphas, name='ok', save=False):
    a = 0
    alpha = r'$ \alpha $'
    for i in arr:
        plt.plot(i,label=f'{alpha}={alphas[a]}')
        a += 1

    plt.title('MSE values for all the tests')
    plt.grid('on')
    plt.xlabel('Iteration number')
    plt.ylabel('Mean square error (MSE)')
    plt.legend(loc=1)
    if save:
        plt.savefig(f'MSE_all_{name}.png', dpi=200)
    else:
        plt.show()
    plt.clf()
    plt.close()
 

#loading the data from the csv file into pandas dataframe

def load_data(path, one=True, maxVal=None, normalize=False, d=','): #if normalise is needed then change this to True
    data = pd.read_csv(path, sep=d)
    if one: #making sure that the data is not just ones
        lenght = len(data)
        #adding ones
        if lenght>60:

            data.insert(4,'Ones',np.ones(lenght),True)
        
        else:
            data['Ones'] = np.ones(lenght)
    #normalize
    if normalize:
        data = data.divide(maxVal)

    return data

#function that removes the feature dataset
def remove_feature_dataset(data, features):
    data = data.drop(columns=features)
    print(data.head())
    return data


# %% [markdown]
# # GLOBAL VARIABLES DECLARATION
# 

# %%

#-------------global variables---------------#
classes = 3
iris_names = ['Setosa', 'Versicolor', 'Virginica']
features = ['sepal_length','sepal_width','petal_length','petal_width']
path = 'iris.csv'
path_setosa = 'class_1.csv'
path_versicolour = 'class_2.csv'
path_virginica = 'class_3.csv'

# %% [markdown]
# # DATA RETRIEVAL 
# 

# %%

tot_data = load_data(path, normalize=False)
max_val = tot_data.max(numeric_only=True).max() #first max, gets max of every feature, second max gets max of the features
setosa = load_data(path_setosa,max_val) 
versicolor = load_data(path_versicolour, max_val)
virginica = load_data(path_virginica, max_val)
#alpha value
alphas = [0.01]



# %% [markdown]
# # PLOTTING FUNCTIONS FOR HISTOGRAM

# %%
def plot_histogram(data): 
    sn.set()
    sn.set_style("white")
    
    # species column is categorical to fix the order of legends
    data['species'] = pd.Categorical(data['species'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for col, ax in zip(data.columns[:4], axs.flat):
        sn.histplot(data=data, x=col, kde=True, hue='species', palette=['red', 'yellow', 'blue'], common_norm=False, legend=ax==axs[0,0], ax=ax)
    plt.tight_layout()
    plt.savefig('newhist_withbestfit.png',dpi=200)
    plt.show()

# %% [markdown]
# # Task 1a
# 

# %%
def task1a(s=True):
    train_size = 30
    arr= []
    features = ['sepal_length','sepal_width','petal_length','petal_width']
    
    
    #splitting the data into train and test
    train = pd.concat([setosa[0:train_size],versicolor[0:train_size],virginica[0:train_size]])
    train_for_test = np.array([setosa[0:train_size],versicolor[0:train_size],virginica[0:train_size]])
    test = np.array([setosa[train_size:],versicolor[train_size:],virginica[train_size:]]) 
    t_k = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]]) 

    train = train.to_numpy()


    #making the model
    for i in range(len(alphas)):
        print(f'Making model with 2000 iteration and an alpha of {alphas[i]} ')
        model = f'w{i}'
        model = LDC(train,test,t_k,2000,alphas[i], features)
        model.train_model()
        model.get_weights()
        arr.append(model.mses)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='test', save=s)
        print('Testing the model with the training set')
        model.reset_cm()
        model.set_test(train_for_test)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='train_1a', save=s)
        

    plot_mses_array(arr, alphas, name='test_1a', save=s)



# %% [markdown]
# # Task 1d

# %%
def task1d(s=True):
    train_size = 20 #can be swapped to 30 if we want the first 30
    arr = [] #
    features = ['sepal_length','sepal_width','petal_length','petal_width']
   

    #splitting the data into train and test
    train = pd.concat([setosa[train_size:],versicolor[train_size:],virginica[train_size:]])
    train_for_test = np.array([setosa[train_size:],versicolor[train_size:],virginica[train_size:]])
    test = np.array([setosa[0:train_size],versicolor[0:train_size],virginica[0:train_size]]) #could mb have done this for train to, 
    t_k = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]]) #making array to check whats the true class is
    #making the data into numpy arrays
    train = train.to_numpy()


    for i in range(len(alphas)):
        print(f'Producing a model with 2000 iterations and alpha values of: {alphas[i]} ')
        model = f'wl{i}'
        model = LDC(train,test,t_k,2000,alphas[i], features)
        model.train_model()
        model.get_weights()
        arr.append(model.mses)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='test', save=s)
        print('Currently training the model with a training set.')
        model.reset_cm()
        model.set_test(train_for_test)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='train_1d', save=s)
        

    plot_mses_array(arr, alphas, name='test_1d', save=s)


# %% [markdown]
# # Task 2a
# 

# %%
def task2a(s=True):

    train_size = 30
    arr = []
    features = ['sepal_length','petal_length','petal_width']
    #removing sepal_width from the features
    re_feature = ['sepal_width']
    setosa1 = remove_feature_dataset(setosa,re_feature)
    versicolor1 = remove_feature_dataset(versicolor,re_feature)
    virginica1 = remove_feature_dataset(virginica,re_feature)


    #splitting the data into train and test
    train = pd.concat([setosa1[0:train_size],versicolor1[0:train_size],virginica1[0:train_size]])
    train_for_test = np.array([setosa1[0:train_size],versicolor1[0:train_size],virginica1[0:train_size]])
    test = np.array([setosa1[train_size:],versicolor1[train_size:],virginica1[train_size:]]) 
    t_k = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]]) #true label array identifier
 

    train = train.to_numpy()


    for i in range(len(alphas)):
        print(f'Producing a model with 2000 iterations and alpha values of: {alphas[i]} ')
        model = f'w2{i}'
        model = LDC(train,test,t_k,2000,alphas[i], features)
        model.train_model()
        model.get_weights()
        arr.append(model.mses)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='test_2a', save=s)
        print('Currently training the model with a training set.')
        model.reset_cm()
        model.set_test(train_for_test)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='train_2a', save=s)
        

    plot_mses_array(arr, alphas, name='test_2a', save=s)


# %% [markdown]
# # Task 2b - 1
# 

# %%
def task2b_1(s=True):


    train_size = 30
    arr = []
    features = ['petal_length','petal_width']
    #removing sepal_width and sepal_length from the features        
    re_feature = ['sepal_length','sepal_width']
    setosa2 = remove_feature_dataset(setosa,re_feature)
    versicolor2 = remove_feature_dataset(versicolor,re_feature)
    virginica2 = remove_feature_dataset(virginica,re_feature)

    #splitting the data into train and test
    train = pd.concat([setosa2[0:train_size],versicolor2[0:train_size],virginica2[0:train_size]])
    train_for_test = np.array([setosa2[0:train_size],versicolor2[0:train_size],virginica2[0:train_size]])
    test = np.array([setosa2[train_size:],versicolor2[train_size:],virginica2[train_size:]])
    t_k = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]]) #true label array identifier
    #making the data into numpy arrays

    train = train.to_numpy()

    for i in range(len(alphas)):
        print(f'Producing a model with 2000 iterations and alpha values of: {alphas[i]} ')
        model = f'w2{i}'
        model = LDC(train,test,t_k,2000,alphas[i], features)
        model.train_model()
        arr.append(model.mses)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='test_2b1', save=s)
        print('Currently training the model with a training set.')
        model.reset_cm()
        model.set_test(train_for_test)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='train_2b1', save=s)
        

    plot_mses_array(arr, alphas, name='test_2b1', save=s)


# %% [markdown]
# # Task 2b-2

# %%
def task2b_2(s=True):

    train_size = 30
    arr = []
    features = ['petal_length']
    #removing sepal_width, sepal_length and petal_width from the features
    re_feature = ['sepal_length','sepal_width','petal_width']
    setosa3 = remove_feature_dataset(setosa,re_feature)
    versicolor3 = remove_feature_dataset(versicolor,re_feature)
    virginica3 = remove_feature_dataset(virginica,re_feature)


    #splitting the data into train and test
    train = pd.concat([setosa3[0:train_size],versicolor3[0:train_size],virginica3[0:train_size]])
    train_for_test = np.array([setosa3[0:train_size],versicolor3[0:train_size],virginica3[0:train_size]])
    test = np.array([setosa3[train_size:],versicolor3[train_size:],virginica3[train_size:]]) 
    t_k = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]]) #true label array identifier

    #making the data into numpy arrays
    train = train.to_numpy()

    for i in range(len(alphas)):
        print(f'Producing a model with 2000 iterations and alpha values of: {alphas[i]} ')
        model = f'w3{i}'
        model = LDC(train,test,t_k,2000,alphas[i], features)
        model.train_model()
        arr.append(model.mses)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='test_2b2', save=s)
        print('Currently training the model with a training set.')
        model.reset_cm()
        model.set_test(train_for_test)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='train_2b2', save=s)
        

    plot_mses_array(arr, alphas, name='test_2b2', save=s)


# %% [markdown]
# # Task 2b-2-1
# 

# %%
def task2b_2_1(s=True):


    train_size = 30
    arr = []
    features = ['petal_width']
    #removing sepal_width, sepal_length and petal_length from the features
    re_feature = ['sepal_length','sepal_width','petal_length']
    setosa4 = remove_feature_dataset(setosa,re_feature)
    versicolor4 = remove_feature_dataset(versicolor,re_feature)
    virginica4 = remove_feature_dataset(virginica,re_feature)


    #splitting the data into train and test
    train = pd.concat([setosa4[0:train_size],versicolor4[0:train_size],virginica4[0:train_size]])
    train_for_test = np.array([setosa4[0:train_size],versicolor4[0:train_size],virginica4[0:train_size]])
    test = np.array([setosa4[train_size:],versicolor4[train_size:],virginica4[train_size:]]) 
    t_k = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]]) 
    #true label array identifier
    train = train.to_numpy()

    for i in range(len(alphas)):
        print(f'Producing a model with 2000 iterations and alpha values of: {alphas[i]} ')
        model = f'w4{i}'
        model = LDC(train,test,t_k,2000,alphas[i], features)
        model.train_model()
        arr.append(model.mses)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='test_2b2_1', save=s)
        print('Currently training the model with a training set.')
        model.reset_cm()
        model.set_test(train_for_test)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='train_2b2_1', save=s)
        

    plot_mses_array(arr, alphas, name='test_2b2_1', save=s)


# %% [markdown]
# # RUN FROM HERE
# Run all cells from the below python cell in order to initialise the Jupyter notebook. After that, one can then simply run any cell below in any order. 

# %% [markdown]
# # HISTOGRAM

# %%
#Runtime code. Run all cells above, then run any cell below.
plot_histogram(tot_data)

# %% [markdown]
# # TASK 1A

# %%
task1a()


# %% [markdown]
# # TASK 1D
# 

# %%
task1d()

# %% [markdown]
# # TASK 2A

# %%
task2a()

# %% [markdown]
# # TASK 2B 1

# %%
task2b_1()

# %% [markdown]
# # TASK 2B 2

# %%
task2b_2()

# %% [markdown]
# # TASK 2B 2-1

# %%
task2b_2_1()



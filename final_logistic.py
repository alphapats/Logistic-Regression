import numpy as np
import warnings

import pandas as pd
#from matplotlib.pyplot import hist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt  
#import scikitplot as skplt
import matplotlib.pyplot as plt
#warnings.filterwarnings( "ignore" )
from sklearn import metrics 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#function to read complete data from given csv file
def read_file(fname):
	data=np.genfromtxt(fname, skip_header=1, usecols=(0,1,2,3),delimiter=',')
	data=data.reshape((len(data),4))
	return(data)


#function to read only attribute data from given csv file
def read_attr(fname):
	data=np.genfromtxt(fname, skip_header=1, usecols=(0,1,2),delimiter=',')
	data=np.array(data)
	return(data)


def normalize(attr): 
    ''' 
    function to normalize feature matrix, X 
    '''
    mins = np.min(attr, axis = 0) 
    maxs = np.max(attr, axis = 0) 
    rng = maxs - mins 
    norm_X = 1 - ((maxs - attr)/rng) 
    return norm_X 


def normalize1(trg_attr): 
    ''' 
    function to normalize feature matrix, X 
    '''
    #scale the age by 10
    trg_attr[:,0] = ((trg_attr[:,0]) / (10))
    new_col = np.multiply((trg_attr[:,0]),(trg_attr[:,2])) 
    trg_attr = np.insert(trg_attr,3, new_col, axis=1)
    #print(trg_attr[:,0])
    calc_mean=np.mean(trg_attr,axis=0)
    dev=np.std(trg_attr,axis=0)
    data=np.copy(trg_attr)
    for row in data:
    	for col in range(0,len(row)):
    		row[col] = (row[col]-calc_mean[col]) / (dev[col])


	return(data)
    


#funtion to read OUTPUT(category) values from csv file
def read_y(fname):
	data=np.genfromtxt(fname, skip_header=1, usecols=(3),delimiter=',')
	data=data.reshape((len(data),1))
	return(data)

#define sigmoid function
def sigmoid(z):
	sig= 1 / (1 + np.exp(-z))
	print(sig)
	return (sig)


def predict( attr,w) :
	z = np.dot(attr,w) 
	Z = sigmoid(z)
	pred_value = np.where(Z >= .500, 1, 0)
	y_pred=   np.squeeze(pred_value)    
	#Y = np.where( Z > 0.5, 1, 0 )

	return y_pred 

def costfn(model_y,y):
	#m=len(phi)
	
	epsilon = 1e-5    
	cost = -y * np.log(model_y+ epsilon) - (1 - y) * np.log(1 - model_y+ epsilon)

 	#cost = np.sum(cost) / m

	return(cost.mean())

def trg(attr, y, step_size, max_iter,conv_condn,lamda):
	phitran=attr.transpose()
	m=len(attr)
	#create intial w (ie w0)
	w=np.zeros((attr.shape[1],1))
	
	num_iter=0
	costs = np.zeros(max_iter)
	converged=False
	z = np.dot(attr, w)
	h = sigmoid(z)
	loss = costfn(h,y)
	while(num_iter<max_iter and converged ==False):
		num_iter=num_iter+1
		old_loss=loss

		z = np.dot(attr,w)
		h = sigmoid(z)
		loss = costfn(h,y)
		costs[num_iter-1]=loss
		#print(loss)
		gradient = ((np.dot(phitran, np.subtract(h, y)) )+ lamda*w )
		#gradient = ((np.matmul(phitran, np.subtract(h, y)) ) )
		#gradient = ((np.matmul(phitran, np.subtract(h, y)) )+ lamda*w ) /m
		gradient=gradient /m
		w=np.subtract(w,np.multiply(step_size,gradient))
		
		change_loss = abs(np.subtract(old_loss,loss))
      	#print(change_loss)
      	if(change_loss<conv_condn):
      		print (num_iter)
      		converged = True
      		return (w,costs)
		#db =   (1/m) * (np.subtract(h, y)) 
		#bias -= step_size * db 
		#bias=np.subtract(bias,np.multiply(step_size,db))
		'''
		z = np.dot(attr, w)
		h = sigmoid(z)
		loss = costfn(h,y)
      	change_loss = old_loss - loss
      	print change_loss
      	if(change_loss< conv_condn):
      		print num_iter
      		converged = True
          '''	
	return (w,costs)

def confusion_matrix_cal(y_pred,test_y):
	# measure performance
	correctly_classified = 0    
	# counter     
	count = 0
	TP=0.0
	TN=0.0
	FP=0.0
	FN=0.0
	precision=0.0
	recall=0.0
	for count in range( np.size( y_pred ) ) :
		#print (y_pred[count], test_y[count])
		if test_y[count] == y_pred[count] :
			correctly_classified = correctly_classified + 1
		if test_y[count] == 1 and y_pred[count] == 1 :
			TP=TP+1
		if test_y[count] == 1 and y_pred[count] == 0 :
			FN=FN+1
		if test_y[count] == 0 and y_pred[count] == 1 :
			FP=FP+1
		if test_y[count] == 0 and y_pred[count] == 0 :
			TN=TN+1
	precision=(TP) / (TP+FP)
	recall = (TP) / (TP+FN)
	Fscore= (2*precision*recall)/(precision+recall)
	accuracy=((TP+TN)/(TP+TN+FP+FN))*100
	return (correctly_classified,TP,FN,FP,TN,precision,recall,Fscore,accuracy)



def make_ROC(y_pred,test_y):
	correctly_classified,TP,FN,FP,TN = confusion_matrix_cal(y_pred,test_y)
	TPR= TP/(TP+FN)
	FPR =FP/(FP+TN)
	# false positive rate
	fpr = []
	# true positive rate
	tpr = []
	plt.figure()
	plt.title('Receiver Operating Characteristic- Logistic')
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	#plt.title("ROC")
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.0])
	x=FPR
	y=TPR
	#plt.plot(x,y)
	plt.plot(FPR,TPR,label="data ")
	
	
	plt.legend(loc=4)
	plt.plot([0,1],[0,1],'r--')
	plt.show()

def plot_roc(y_test, y_pred):
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()



#read and split data into test and train
input_file = read_file ("health_data.csv")
np.random.shuffle(input_file )

indices = range(input_file.shape[0])
num_training = int(0.8 * input_file.shape[0])
num_test = int(input_file.shape[0]) - num_training
print("Training data",num_training)
print("Test data",num_test)
train_indices = indices[:num_training]
test_indices = indices[num_training:]



# split the actual data
train_x, train_y = input_file[train_indices,0:3], input_file[train_indices,3:]
test_x, test_y = input_file[test_indices,0:3], input_file[test_indices,3:]



'''

#read attributes of data
data_attr=read_attr("health_data.csv")
#print (data_attr)
#phi=normalize(data_attr)
phi=normalize1(data_attr)

#read output 
data_y=read_y("health_data.csv")
#print (data_y)
y=data_y
'''

#setting parameters
step_size = 0.002 # step size
conv_condn=0.001 #condition for convergence
max_iter=2000#maximum number of iterations
lamda=0.0005 #value of lambda

train_x=normalize1(train_x)
w_final,costs= trg(train_x, train_y, step_size, max_iter, conv_condn,lamda)

plt.plot(costs)
plt.xlabel('Number Epochs'); plt.ylabel('Cost');
plt.show()

# predicted labels 
test_x=normalize1(test_x)
y_pred = predict(test_x, w_final) 
#print(y_pred)     

# measure performance

correctly_classified,TP,FN,FP,TN,precision,recall,Fscore,accuracy = confusion_matrix_cal(y_pred,test_y)

m=len(test_y)

print("Correctly predicted values:", correctly_classified , " from ",  m ," cases")
#print("Accuracy: ", (correctly_classified* 100)/m )

print("Accuracy: ", accuracy)
print("Precision : ",precision)
print("Recall : ",recall)
print("Fscore : " ,Fscore)

print( confusion_matrix(test_y ,y_pred))
print ("True Positive", TP)
print ("False Positive", FP)
print ("False Negative", FN)
print ("True Negative", TN)


conf_matrix=[[TN, FP],[FN, TP]]
print(conf_matrix)

TPR= TP/(TP+FN)
FPR =FP/(FP+TN)
#make_ROC(test_y,y_pred)
plot_roc(test_y,y_pred)
#skplt.metrics.plot_roc_curve(test_y, y_pred)
#plt.show()



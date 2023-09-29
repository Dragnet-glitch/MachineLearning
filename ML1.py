#Here dependencies are imported
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Here data is loaded to the pandas data frame, The data is called sonar_data as a variable
sonar_data = pd.read_csv("sonar.csv", header=None)
#here we check the first 5 rows and columns of the data
#print(sonar_data.head())#uncomment print(sonar_data.head()) to check.
#here we check the shape of the data
#print(sonar_data.shape)#uncomment print(sonar_data.shape)  to check that data shape.
#here we can check the mean and standard deviation of the data
#print(sonar_data.describe())....#uncomment print(sonar_data.decribe()) to check the mean and standard deviation of the data
#here we can see the number of outcomes that is mines and rocks
#print(sonar_data[60].value_counts())#uncomment it to see them
#now we need to seperate the data and labels
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]
#print(x)
#print(y) #uncomment to check if they are seperated
#now we split the data into training and test data.
#0.6 means reserving 60% for testing.You can adjust as u wish
#we stratify the y variable which is the outcome so that we can have a fair amount of training.
#the random state is also used to maintain consistency
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)
#now let us see the shape of our x train and x test.
#You will notice they are balanced for both x and y
#print(x.shape, x_train.shape, x_test.shape)
#print(y.shape, y_train.shape, y_test.shape)
#finally our model
model = LogisticRegression()
#now we fit our x and y train into the data while excluding the test data.
#we do not want our model to see the test data
model.fit(x_train, y_train)
#lets test the data based on the trained data
x_train_prediction = model.predict(x_train)
data_accuracy = accuracy_score(x_train_prediction, y_train)
#print('Accuracy on the trained data is : ', data_accuracy) #uncomment to see the accuracy of the trained data.
#lets test the data based on the test data
x_test_prediction = model.predict(x_test)
data_accuracy = accuracy_score(x_test_prediction, y_test)
#print('Accuracy on the test data is: ', data_accuracy) #uncomment to see the accuracy of the test data.
#here we input the raw data from the sonar.csv file 
input_data = (0.0209,0.0261,0.0120,0.0768,0.1064,0.1680,0.3016,0.3460,0.3314,0.4125,0.3943,0.1334,0.4622,0.9970,0.9137,0.8292,0.6994,0.7825,0.8789,0.8501,0.8920,0.9473,1.0000,0.8975,0.7806,0.8321,0.6502,0.4548,0.4732,0.3391,0.2747,0.0978,0.0477,0.1403,0.1834,0.2148,0.1271,0.1912,0.3391,0.3444,0.2369,0.1195,0.2665,0.2587,0.1393,0.1083,0.1383,0.1321,0.1069,0.0325,0.0316,0.0057,0.0159,0.0085,0.0372,0.0101,0.0127,0.0288,0.0129,0.0023)
#convert the input dara to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
#we reshape the numpy_array_input_data
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
#the prediction varaible uses the model to predict based on the input
prediction = model.predict(input_data_reshaped)
print(prediction)
#if the prediction is R, it prints rock 
#if the prediction is M, it prints Mine
if prediction == 'R':
    print("rock")
if prediction == 'M':
    print("Mine")


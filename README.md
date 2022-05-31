## Digit Recognizer: Keras + PyTorch

### 1. Introduction

For this Digit Recognizer Kaggle project, we wanted to use more than one model for this machine learning project as well as create some visualizations. We decided to use the Keras and Pytorch models. For both of these models, we researched ways in which we can increase the accuracy in order to have the best working models. The models we’ve created are what we believe to be the best ways we were able to approach the problem and increase accuracy based on our research. <br>

The dataset can be downloaded through Kaggle compepition. <br>
Below is the link: https://www.kaggle.com/competitions/digit-recognizer/overview

### 2. Data Preprocessing

We first import required packages including pandas, math, numpy, matplotlib, random, tensorflow, torch, and sklearn. Then we read the “train.csv” and  the “test.csv” files from Kaggle as our training and test dataset. We create two files from "sample_submission.csv" as our submission file for Keras and PyTorch separately. The final step was splitting the train and test data into input and output values and reshaping the data in two dimensions (height and width).

### 3. Exploratory Data Analysis 

For the data exploratory, we were curious about the distribution of digits in the training data and found out that the number of each digit is very similar to each other in the training dataset which is indicated in the bar chart shown below.

![](https://ppt.cc/fd0Qax@.png)

We then visualized the digits into pictures and let the user choose how many rows of training images they would like to review. The figure below shows that.

![](https://ppt.cc/fL9qGx@.png)

### 4. Keras Model

-	**Results and Reports** <br>
From doing the Keras model, we received high accuracy. We were able to increase both the training and validation accuracies from the basic model given in class. The training accuracy is over 98% and the validation accuracy is over 97%. 

-	**Methods** <br>
We researched different ways in which we can improve the accuracy in our models. The approaches we tried in order to improve accuracy include adding more layers, increasing the dense layer relu size, increasing the dense layer softmax size, changing the epochs, changing the batch size, changing the validation split, L1 regularization, L2 regularization, dropout, and batch normalization. The best approach that increased the accuracy the most was increasing the dense layer relu size, increasing the dense layer softmax size, changing the epochs to 80, and changing the batch size to 80. These four changes increased the accuracy of the Keras model more than the other approaches that were tried. 

### 5. PyTorch Model

-	**Data Preparation** <br>
We first split the data into training and validation datasets. Then, we scaled the values between 0 and 1 for both the training and testing sets and transformed the data into tensors. We set the batch size to 20 and the shuffled to False.

    We created the model with only one hidden layer with a hidden size of 500 and applied linear transformation to hidden = input A^T + b as well as output     = hidden A^T + b. Finally, we applied the dimensional output Tensor within the range [0,1] and summed to 1.

-	**Train Model** <br>
For training the model, we set the learning rate to 0.001 and implemented the Adam algorithm as the optimizer. We set the epochs to 50 and got the highest training accuracy of around 0.9968.

-	**Evaluate Model Performance** <br>
Then, we continued calculating the accuracy for the validation data set and got the accuracy to around 0.9754.

-	**Prediction of Test Dataset** <br>
Finally, for predicting the test data, we first created an empty list to record the label prediction. We then transformed the list to one dimension. After printing the result into the list, we read the total prediction into the submission file to submit.

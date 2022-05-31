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

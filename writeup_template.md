# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image-hist-before-augmentation]: ./output_images/data-visualization-before-augmentation.PNG "Classes histogram before augmentation"
[image-hist-after-augmentation]: ./output_images/data-visualization-after-augmentation.PNG "Classes histogram after augmentation"
[image-rotate]: ./output_images/rotate_img_num_1.png "Rotated image"
[image-shift]: ./output_images/shift_img_num_1.png "Shifted image"
[test-image1]: ./test_images/Traffic_Sign_01.png "Test Traffic Sign 1"
[test-image2]: ./test_images/Traffic_Sign_02.png "Test Traffic Sign 2"
[test-image3]: ./test_images/Traffic_Sign_03.png "Test Traffic Sign 3"
[test-image4]: ./test_images/Traffic_Sign_04.png "Test Traffic Sign 4"
[test-image5]: ./test_images/Traffic_Sign_05.png "Test Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 examples
* The size of the validation set is 4410 example
* The size of test set is 12630 example
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are distribured among the traffic sign classifier classes.

![alt text][image-hist-before-augmentation]

Here is an exploratory visualization of the data set after augmentation that will be discussed in details below.

![alt text][image-hist-after-augmentation]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it makes the performance better, and there is no important information to be extracted from a colored image.

Second, I apply histogram equalization for the images so that the grey levels for an image are destributed so that a very dark image looks priter, and a very bright image looks darker.

As a last step, I normalized the image data so that the overall mean is zero, and it also makes interpolation easier.

I decided to generate additional data because the data distribution as shown in the histogram for the traffic sign classifier classes is very unfair.
We need the number of sample for each class to by high enogh in comparison with other classes to avoid the dominance for one class above the others in the training phase.

To add more data to the the data set, I used the following techniques:
 * Rotate the image randomly between 3, and 15 degrees.
 * Translate the image randomly between 2, 6 in both x, and y directions.
 * Rotate and translate randomly for the same image.
The algorithm for augmenting the data to the data set goes as follows:
```
foreach a-class in all-classes
 if a-class.number-of-examples < 1000
  randomly-shift-and-rotate-the-image
  if a-class.number-of-examples < 400
   randomly-shift-the-image
  if aclass.number-of-examples < 600
   randomly-rotate-the-image
  if aclass.number-of-examples < 900
   randomly-shift-and-rotate-the-image
```

Here is an example of an original image and an augmented rotated image:

![alt text][image-rotate]

Here is an example of an original image and an augmented shifted image:

![alt text][image-shift]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Layer 1: Convolutional         		| The input is 32x32x1, and the output shape should be 28x28x16 using relu activation function. |  			| Layer 2: Convolutional | The input is 28x28x16, and the output shape should be 28x28x32 using relu activation function. |
| Max pooling     	| 2x2 stride, same padding, outputs 14x14x32 	|
| Drop out					|		Keep probability = 0.9										|
| Layer 3: Convolutional	      	| The input is 14x13x32, and the output shape should be 10x10x64 using relu activation function				|
| Layer 4: Convolutional	    | The input is 10x10x64, and the output shape should be 10x10x128 using relu activation function   | 							| Max pooling | 2x2 stride, same padding, outputs 5x5x128 |
| Drop out		| Keep probability = 0.8       									|
| Flatten			| Flatten the output shape (3200) of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported.        									|
|		Layer 5: Fully Connected				|			This should have 2500 outputs with relu activation function.								|
| Drop out		| Keep probability = 0.7       									|
|		Layer 6: Fully Connected				|			This should have 2000 outputs with relu activation function.								|
| Drop out		| Keep probability = 0.6       									|
|		Layer 7: Fully Connected				|			This should have 1000 outputs with relu activation function.								|
| Drop out		| Keep probability = `variable-passed-to-lenet-architecture`      									|
|		Layer 8: Fully Connected				|			This should have 500 outputs with relu activation function.								|
| Layer 9: Fully Connected			| (Logits). This should have 43 outputs that represents all the traffic sign classifier classes.      					|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a learning rate of 0.001, 20 epochs, batch-size of 128, and the tensorflow architecture discussed above.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98%
* validation set accuracy of 96%
* test set accuracy of 93.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? Lenet architecture.
* What were some problems with the initial architecture? The overall accuracy was 89% which is lower that expected.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why? Lower learning rate to reach better global minimum, and the drop-out to help model generalization.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? The convlution works very well with image application because it is not relevant to the translation, and the location for an object in the image. It also shows a very good accuracy for image applications.

If a well known architecture was chosen:
* What architecture was chosen? Lenet architecture.
* Why did you believe it would be relevant to the traffic sign application? Because the Lenet architecture shows a very good performance on image applications, and it is not relevant to the location, and the translation for an object in the image.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? When used this model before any adjustements it reaches 89% accuracy, and after preprocessing, data set augmentation, and adding more layers to the model as discussed above in the architecture, I reached 96% accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test-image1] ![alt text][test-image2] ![alt text][test-image3] 
![alt text][test-image4] ![alt text][test-image5]

The first image might be difficult to classify because the number of training data associated with its class is very low.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve     		| Right-of-way at the next intersection  									| 
| Keep left     			| Keep left										|
| No entry					| Turn left ahead											|
| Roundabout mandatory	      		| Roundabout mandatory				 				|
| Turn left ahead			| Turn left ahead      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. But in the juptyer notbook I tried on 15 images and I got 70% overall accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 31th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Right-of-way at the next intersection (probability of 0.9779668), and the image doesnot contain a Right-of-way at the next intersection. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9779668        			| Right-of-way at the next intersection   									| 
| 0.021402394     				| Beware of ice/snow 										|
| 0.00026011598					| Dangerous curve to the right											|
| 0.00018698753      			| Children crossing					 				|
| 7.128884e-05				    | Pedestrians       							|


For the second image, the model is relatively sure that this is a Keep left (probability of 1.0), and the image does contain a keep left. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Keep left   									| 
| 0.0     				| Speed limit (20km/h) 										|
| 0.0					| Speed limit (30km/h)											|
| 0.0      			| Speed limit (50km/h)					 				|
| 0.0				    | Speed limit (60km/h)       							|

For the forth image, the model is relatively sure that this is a Roundabout mandatory (probability of 1.0), and the image does contain a Roundabout mandatory. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Roundabout mandatory   									| 
| 2.2574362e-08     				| Priority road 										|
| 2.3476138e-11					| Speed limit (30km/h)											|
| 2.2616863e-11      			| Speed limit (100km/h)					 				|
| 2.4419957e-13			    | Bicycles crossing       							|




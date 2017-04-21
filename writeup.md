#**Behavioral Cloning** 

##Writeup Template

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a neural network with single convolution with a 5x5 filter size 
and depth of 6 (model.py lines ~177) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

The data has also been cropped to (60, 20), (0, 0) using a Keras lambda layer.

####2. Attempts to reduce overfitting in the model

The model contains a max pooling layer in order to reduce overfitting (model.py lines 184) and was only trained for 2 epochs.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 198). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 190).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 

I only used center lane driving to train the model, as using recovery driving cannot be simulated in real life so I didn't see the value in it.

I used right and left camera angles to help train the model to recover from off center driving.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep it simple. 

My first step was to use a convolution neural network model similar to LeNet, however with just 1
convolution.

I thought this model might be appropriate because, unlike image classification where the model needs to go through layers of detail to correctly classify the images, the goal of this task was to stay on the track. To do that the model needed to understand what constitutes each edge of the track, which is a much simpler task than understanding the features of an animal, for example.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80/20). 

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it only ran on 2 epochs and introduced max pooling (line 191).

The final step was to run the simulator to see how well the car was driving around track one. 

There were a few spots where the vehicle fell off the track, in particular near the dusty edge, so I collected more data around this area to reinforce the correct path to take.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 184-195) consisted of a convolution neural network with the following layers and layer sizes:

| Layer (type)                   | Output Shape        | Param # |
|--------------------------------|---------------------|---------|
| lambda_1 (Lambda)              | (None, 160, 320, 3) | 0       |
| cropping2d_1 (Cropping2D)      | (None, 80, 320, 3)  | 0       |
| conv2d_1 (Conv2D)              | (None, 76, 316, 6)  | 456     |
| max_pooling2d_1 (MaxPooling2D) | (None, 38, 158, 6)  | 0       |
| activation_1 (Activation)      | (None, 38, 158, 6)  | 0       |
| flatten_1 (Flatten)            | (None, 36024)       | 0       |
| dense_1 (Dense)                | (None, 1)           | 36025   |

Total params: 36,481

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded 3 laps on track one using center lane driving. 

Here is an example image of center lane driving:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Behavioral-Cloning-P3/master/examples/center_camera.jpg)

To augment the data sat, I also flipped images to remove the anti-clockwise bias from the training circuit, flipping the corresponding steering angle.

To help centre road driving and streeing correction I also used the left and right camera angle images captured when training, adjusting the throttle angle by 0.2 in the positive direction for the left camera angle, and in the negative direction for the right camera angle.

An example of *left* camera angle:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Behavioral-Cloning-P3/master/examples/left_camera.jpg)
Original Steering Angle: 0. After Adjustment: 0.2

An example of *right* camera angle:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Behavioral-Cloning-P3/master/examples/right_camera.jpg)
Original Steering Angle: 0. After Adjustment: -0.2
 
After the collection process I had 73,998 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped 
determine if the model was over or under fitting. The ideal number of epochs 
was 2 as evidenced by a rising validation error 3 epochs.
 
I used an adam optimizer so that manually training the learning rate wasn't necessary.

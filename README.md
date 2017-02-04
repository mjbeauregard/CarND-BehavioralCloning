# Project 3: Behavioral Cloning

This is my submission for "Project 3: Behavioral Cloning". This project involves using deep learning to train a CNN to drive a car around the track in a driving simulator. Human driver behavior is captured via the simulator as a series of images and steering angles. From this information, the CNN must successfully learn to drive around the track by cloning the human driving behavior.

## Capturing Training Data

It is actually pretty tricky to capture good data using the simulator for a couple reasons. The first issue with capturing training data is that the keyboard-based simulator captues steering angles that result in over controlling the car since it either registers no steering at all (i.e. 0 when no keys are pressed) or large steering angles as soon as you press a key. Second, the "beta" simulator that provides mouse control for smoother steering doesn't generate left and right camera images which have proven to be critical for keeping the car in the center of the lane. 

Because of these challenges with capturing good data, I had to rely on the sample training data provided by Udacity  - it looks like that data was captured using a joystick (which I do not have). The sample training data is limited to about 8000 frames which ended up being enough to successfully train the network.

## Preprocessing/Augmentation

The input csv driver log meta-data was fully loaded into memory with telemetry and image file paths. Because the image data is much to large to fit into memory, a generator was used to feed the training processing by incrementally loading batches of randomly selected data. Only one full batch of image data was loaded at any time (batch size is 256). 

The generator loads data with the following procedure:
1. randomly selects record from the data set
2. randomly selects one of the camera images (left, center, right)
3. crops the selected image to exclude most of the image above the horizon and the bottom part of the image to remove the hood of the car
4. downsamples the image to 1/5th in each dimension - the resulting image shape was (21, 64, 3)
5. randomly flips the selected image and steering angle to help reduce the left-turn bias of the training data

If either the left or right camera image was selected, the input steering angle was corrected by subtracting or adding 0.11 from the recorded (center) steering angle. The approach ultimately seemed to help provide some pressure to keep the car from drifting around the road and kept it much closer to the center.

The track consists mostly of left turns which results in training data that is heavily biased toward turning left. To combat this situation, the generator would randomly flip the input image along the vertical axis. The result is a better mix of left and right turn images.

Finally, it's interesting to point out that one of the most important factors in successfully getting all the way around the course was step #4 where the image was significantly downsampled. Perhaps the higher resolution data is too noisy and distracting for the network to successfully learn to control the car.

## Model Architecture

This project uses a CCN model architecture similar to the Nvidia paper (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The model takes an RGB image as input and produces the steering angle as the only output.

The input layer of the model normalizes the image data to the range [-1, 1]. This layer is then followed by a series of 5 convolutional layers. Finally, the convolutional output is flattened and passed through a series of fully connected layers successively reducing the number of outputs, ultimately resulting in a single real value output as the predicted steering angle. The model uses "relu" activation throughout. 

The total number of parameters of the network is 1,148,535 which seems high compared to the original Nvidia model, but subsampling wasn't possible on the first few convolutional layers once the input image had been downsampled as far as it was.

Various combinations of other convolutional, dense and dropout layers were experimented with, but none of them ever produced any improvements so were ultimately discarded from the final model. It's awesome how easy it is to define a network using Keras and trivially experiment with different ideas as mentioned.

The following is a summary of the network architecture:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 21, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 17, 60, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 13, 56, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 9, 52, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 25, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 12, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 768)           0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          895116      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================

## Training

The training process was way too slow to perform on a laptop cpu so I opted to leverage an AWS g2.8xlarge instance to speed up the iterative process. Due to difficulties with creating quality training data, I ended up entirely relying on the Udacity sample training data that was provided.

Using simple augmentation the generator produces 6 samples (3 camera angles x 2 random flip) per recorded training sample. I reserved 20% of the data for validation, though this is probably overkill since validaiton accuracy was not very indicative of how well the network would actually perfom in the simulator. Nonetheless, the loss would always settle down within the alotted 10 epochs. The model was trained using the "adam" optimizer using "mse" to calculate loss.

## Results

It was difficult to get the car to successfully drive all the way around the test track until the input images were drastically scaled down. As mentioned, many iterations of different network layers were attempted (more/different conv layers as well as dropout and batch normalization) but generally made little improvement and often worse perforance.

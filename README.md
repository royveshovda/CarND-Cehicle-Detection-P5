# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I have provided this writeup as part of a GitHub-repo, so this is the README you are reading.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code is contained in the file called 'extract_features.py' in the function 'get_hog_features'. I started by loading all the files, both vehicles and non-vehicles. The loading happens in the file 'load.py', and all the images are saved as numpy dat-files, after they are resized to 64x64. This allowed me to play around with different parameters without having to run the loading and resizing again and again.

After I have extracted all the features (I chose to extract histogram features as well as spatial features, in addition to HOG features), I saved the features for all the images back to numpy dat-files.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and ended up using the colorspace for YCrCb and all the channels. Simply because this seems to give the best result after training.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM from sklearn by using the features extracted from the previous step, and the code in the file 'train.py'.

I first applied a scaling step to normalize the input features. Then I split the data in 80% training data, and 20% test data. Before I called the train function. After the classifier was done training, I saved the classifier and scaler using pickle. This allowed me to play around with different pipelines without having to retrain the classifier.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

## TODO

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

## TODO

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out_project.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

## TODO

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem with this implementation is speed. Is is way too slow to do near realtime detection on a video stream. The runtime for a single image is about 2 seconds. So several optimizations should be considered.

I use various scalings to search for cars. This search could be varying the search space based on the scale. As mentioned in class, small cars will only appear at a distance, so we should only search for them at a distance. The current implementation executes the search for all scales inside all the y-boundaries. This could speed up the process.

The detection in itself seems to do a pretty good job on the provided videos. The classifier needs to be trained on a much larger image base and under various light conditions to be useful in real life scenarios.

However the biggest improvement could come from implementing a Deep Learning pipeline. Convolutional Neural Networks have come a long way these days, and should be able to solve this project fairly easy.

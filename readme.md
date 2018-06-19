# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: vehicle_feature.png
[image2]: non_vehicle_feature.png

[image3]: test1_proc.png
[image4]: test2_proc.png
[image5]: test3_proc.png
[image6]: test4_proc.png
[image7]: test5_proc.png
[image8]: test6_proc.png

[video1]: project_video_proc.mp4

---

### 1. Feature Extraction

The code for this step is to read the training image.  All training image are re-sized as 64 by 64.  After that, the image features are extracted.  The following features are used in this project  

#### 1.1 Color Space

RGB color space could give some information about the vehicle or non-vehicle.  However, RGB color space would not provide reliable information for next classifier.  Therefore, the YUV color space is used.  All the following features are from the YUV color space.  
 
#### 1.2 Histogram of Oriented Gradients (HOG)

HOG feature is the main feature used to classy vehicle from the background.  The HOG is computed over all three channel of the YUV color space.  The scikit-image hog() function is used to extract HOG. The key parameters include orientations, pixels_per_cell and cells_per_block.  In this project, HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` are used.  These parameters are commonly used, and provide an acceptable performance in general.

There are two examples.  One is for the vehicle image.  Another is for the non-vehicle image.

##### Vehicle Image
![alt text][image1]

##### Non-Vehicle Image
![alt text][image2]

#### 1.3 S YUV Channel Histogram

All three channel histogram is used to compose the feature space since all of them provides useful information. 

#### 1.4 Spatial Features

The raw pixel values are still quite useful to include in the feature vector in searching for vehicles.  The spatial features are extracted as a 32 by 32 Y channel.  The reason of not using all three channel is to reduce the feature length.

---

### 2. SVM Classifier

#### 2.1 Linear SVM 

Sk-learn SVM implementation is used in this project.  The linear SVM is chosen to reduce the computation loads.  A small C parameter is chosen to reduce over-fitting effects.  The training and test scores are very good.  This SVM classifier has the training accuracy of 1.0 and testing accuracy of 0.9916.  The actual detection on the test image are not that good.  It is clear that this linear SVM suffers from over-fitting. 

#### 2.2 decision_function 

In order to reduce the over-fitting problem, the decision function is chosen instead of predict function.  The benefits of decision_function is to output the score of the classifier instead of the classification results as the predict.  This score information is used in the later heat map computation to improve the performance. 

---

### 3. Static Image Pipeline

#### 3.1 Sliding Window Search

In order to search vehicles in the image, the slide window method is used.  Since the vehicle could be large or small depending on the distance, the multiple scale windows have to used.  However, brutal force search over the entire image using the multiple scale window could cost huge amount of computation power.  Thus, the following two points are used in the project to greatly reduce the computation costs

* Only search the image portion which could have the detectable vehicle, which are the lower part of the image.
* Small window search only applied to the center of the image, where the distance is far, and the vehicle size is small. 

#### 3.2 HOG computation

In order to reduce the computation load further more, extract HOG features just once for the entire region of interest (i.e. lower half of each frame of video) and subsample that array for each sliding window.  Refer the code for more details.

#### 3.3 Multiple detection and False Positive Handling

Since the multiple scale sliding window is used, a vehicle might be detected multiple times by the different windows.  Thus, all detections are being integrated to generate a heat map.  Then simply threshold the heatmap to remove false positives.  Then use `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  Assumed each blob corresponded to a vehicle, and construct bounding boxes to cover the area of each blob detected.  


#### 3.4 Examples

Here are six test image results.  One of them does not have vehicle.  These results clearly show the success of the proposed pipleline to detect the vehicles.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

---

### 4. Video Implementation

#### 4.1 Heating Map Implementation
The key difference between the video and the static single image is that the successive frame in video could have much similarity, and vehicle shown in one frame would be very likely shown in the next frame.  Thus, a exponential average of the heat map is used to explore this property.

#### 4.2 The Results

Here's a [link to my video result](project_video_proc.mp4).  In this vidoe, it is clear that the proposed video pipeline is able to detect the vehicles most times even though there are a couple of false positive happens, which are not removed by the heatmap.  Overall, the performance is acceptable even though there are still some rooms for further improvements.

---

### 5. Discussion

The propose image and video pipeline both show good performance.  However, these implementations are far away from perfection.  The further improvements could be made on the following areas

* Deep study HOG scheme, and further optimize the HOG features.
* Increase the training sample, particular some training samples whose representations are close to the final applications
* Further optimize the SVM parameters, including kernel, C parameter, and gamma.
* Study the sliding window implementation, and decide the best trade off between the performance and the computation costs.
 


# Classification and Detection of Singapore Road Traffic Signs

## Background
In the global automotive industry today, self-driving cars, also known as autonomous vehicles, have been one of the key innovations that is brought about by accelerating adoption of artificial intelligence and robotics. An [article](https://apnews.com/press-release/Wired%2520Release/79c308d2e72d77a9a755be454b3a278a) by The Associated Press wrote that according to Allied Market Research, the global autonomous vehicle market is estimated to garner $556.67 billion by 2026 with a compound annual growth rate of 39.4% during period 2019–2026. Furthermore, [Consultancy Asia](https://www.consultancy.asia/news/3382/singapore-is-the-globes-top-country-for-autonomous-driving) wrote that KPMG released a benchmark report in mid-2020, naming Singapore as the top country in the world in terms of development of self-driving cars, as reflected in the government's policy-making and legislation efforts to encourage use of autonomous vehicles.

Not only is an AV expected to transport people and goods efficiently and safely from point-to-point, it must also comply with existing road regulations as what is expected of human drivers. One of the primary indicators of road regulations and information would be traffic signs. In Singapore, they range from the green directional signs that tell drivers about upcoming expressway exits, to warning signs that indicate potential elements of danger ahead, such as pedestrian crossings.

## Problem Statement
This project is focused on the development both a classification model and an object-detection model to recognise road traffic signs in Singapore. While there are established datasets on traffic signs for countries like Germany and the U.S., there has not been one that is readily available and applicable to the signs found in Singapore. Hence there is a need to build a dataset of local traffic signs as a start, which would be sufficient to facilitate analysis, as well as aid the construction of classification and detection models that could recognise, as well as locate 1 or multiple traffic signs on a given image.

## Executive Summary

### Classification
- For the traffic sign classification dataset, it had 2895 images, across 32 classes of traffic signs.
![32 traffic signs for classification](Images/classification_32_signs.jpg)

- Exploratory Data Analysis revealed a class imbalance, with *WARNING_SLOW_SPEED* as the largest class having 227 images, while *PROHIBITORY_NO_LEFT_TURN* is the small class with only has 26 images. While the images are all cropped into square format, there is a still a variance in width, with most images measuring around 120-140 pixels at each side. Mean brightness is slightly below average, owing to factors like exposure compensation in videos, as well as real-world conditions such as degree of shade or exposure to sunlight. Perspective variance arises from factors such as mounting of traffic signs, direction in which they are facing in relation to camera, and lens distortions in video capture device. Other variances include faded colours on signs, obstructed view by foreign objects, or simply dirt on signs. Colour cast due to video capture device, external weather elements like rain, and night-time street lamp illumination will also affect how traffic signs appear to viewers. Howver, despite all these variances, average-image analysis revealed that most traffic signs are still discernable from one another, with reasons including sign being mostly front and center in the image, and inherent design of signs being very distinguishable in the first place.


- In the pre-processing phase, we use the image data-generator features of Tensorflow to perform data augmentation, as well as train/validation split. A test set was introduced, containing traffic signs with off-centre position, more background clutter and other variances. This would serve as a stringent and impartial evaluation of the trained model, apart from the validation set that steers the minimising of the loss function during model training.


- Tensorflow 2 Keras framework was used to construct the neural network models here. 1st model was a basic convolutional neural network consisting of 1 Conv2D layer with 32 filters, connected to dense hidden layer of 32 units, then followed by output dense layer of 32 units to reflect the categorical classification of the 32 classes. Test accuracy was only 60.8%, evening though training accuracy topped out at 99.6%, with validation accuracy at 88.0%. Validation loss and accuracy plots stabilised after around 10 epochs with no deterioration, suggesting that insufficient network capacity was more of an issue than overfitting.


- 2nd model was constructed with 4 Conv2D layers with accompanying max-pooling layers to down-sample the feature map outputs from respective convolutional layers. This is followed by 3 more hidden dense layers with accompanying dropout layers as a form of regularisation to mitigate overfitting tendencies of a deep network. Early stop criteria was set to monitor training loss for stagnant over 5 epoch, but was never triggered. This time, train accuracy was 99.9%, validation at 97.3%, and **test accuracy at 90%**. All 3 accuracy metrics are well within 10% deviation of each other, and progress of validation loss and accuracy optimisation were mostly keeping pace with the training progress throughout the 50 epochs.


- Inspection of incorrect predictions on test set revealed that model was misclassified *MANDATORY_TURN_LEFT* sign as *MANDATORY_KEEP_LEFT* sign, owing to a slight tilt in the test image, as well as close similarities between both signs. This underscores the need to evaluate data augmentations carefully, and avoid using the options that might introduce ambiguity between classes. Misclassifications between *WARNING_MERGE* with *WARNING_ERP* and *WARNING_RESTRICTED_ZONE_AHEAD* indicate that the model might have insufficient training in differentiating between these warning signs, especially when the pre-dominant similarity is the red triangle on white background. Statistics on class distribution indicate that all these 3 signs have under a hundred images, hence the lack of sufficient data might have affected the model's ability to fully learn the differences in their patterns.

### Object Detection
- For the Object Detection dataset, it originally had 2554 annotations performed on 1560 images, across 57 classes of traffic signs.


- Exploratory data analysis revealed that while most images have 1 traffic sign present, approximately one-third of the others have at least 2 traffic signs in them. Most annotated traffic signs occupy only up to 0.5% of the image area, highlighting the challenge of small object detection, where signal-to-noise ratio is really low. Scatterplot analysis revealed that most traffic signs occur at the left and right sides of the a given frame, in middle-third range in terms of frame height. Largest signs tend to be the overhead directional signs typically found on expressways.


- Due to class imbalance, we decided to focus on 7 traffic signs that have at least 100 obervations. They are *Directional Sign*,*Mandatory Split Way*, *Prohibitory No Jaywalking*, *Prohibitory No Veh Over Height 4.5m*, *Temp Work Zone Sign*, *Warning Curve Right Alignment Marker*, and *Warning Slow Speed*.

![7 traffic signs for object detection](Images/od_7_signs.jpg)

- In pre-processing stage, Object Detection dataset had to be filtered for images containing at least 1 of the 7 picked classes, resulting in 767 images remaining for modeling phase. Their XML annotations were traversed to remove other classes of traffic signs that were originally annotated. Multi-label train/validation/test split was performed on dataset. Images and annotations were converted into TFRecord format that would be used for the model training and evaluation process. Project assets were transferred to Google Drive location.


- Model training was carried out on Google Colab environment aided by GPU-acceleration, using Tensorflow Object Detection API. Training duration was approximately 3 hours, with overall loss reaching 0.093. Trained model was evaluated to have a **mAP score of 0.758**. Inference on 76 images from *test* set was completed with a mean inference time of 45ms per image.


- The *test* set inference images revealed that model was able to detect small traffic signs and large directional signs quite accurately. False positives surfaced in the form of untrained traffic signs being identified as at least 1 of the 7 classes the model was trained for. There were a few false negatives for small traffic signs, indicating there was room to imrpove the recall of the model.

# Classification Model Performance Table

| Model  | Train Acc. | Validation Acc.  | Test Acc. |
|-|-|-|-|
| model_1   | 99.6% | 88.0% | 60.8%  |
| model_2   | 99.9% | 97.3% | 90.6%  |

# Object Detection Model Performance Table
Based on transfer-learning of 7 traffic signs on `ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8` with pre-trained weights.
| Metric | Area | Dets | Score
|-|-|-|-|
|Average Precision  (AP) @[ IoU=0.50:0.95|area=all|maxDets=100|**0.758**|
|Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 |0.924|
|Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 | 0.880|
|Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 | 0.300|
|Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 | 0.787|
|Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 | 0.708|
|Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 | 0.717|
|Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 | 0.814|
|Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 | 0.814|
|Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 | 0.600|
|Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 | 0.818|
|Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 | 0.867|

# Object Detection Inference samples on Test Set

Red bounding box represents ground-truth.

![01](Images/test_set_object_detection_predictions/24_s_008260_with_prediction_48.jpg)

![01](Images/test_set_object_detection_predictions/24_s_036240_with_prediction_40.jpg)

![01](Images/test_set_object_detection_predictions/24_s_083500_with_prediction_41.jpg)

![01](Images/test_set_object_detection_predictions/28_s_000620_with_prediction_46.jpg)

![01](Images/test_set_object_detection_predictions/25_s_005500_with_prediction_41.jpg)

![01](Images/test_set_object_detection_predictions/24_s_002950_with_prediction_44.jpg)

> Written with [StackEdit](https://stackedit.io/).

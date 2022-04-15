# Variable Message Sign Extraction and Classification

_Overview:_
There are an increasing number of partially autonomous vehicles (AV) on the road which require the most up to date information on the environment for appropriate and safe decision making.
Temporarily relevant information regarding road conditions, detours, traffic conditions, and closures are
often displayed on Variable Message Signs (VMS) on the side of roads. This information can be critical
for AV to react appropriately in each situation, which then gives rise to a need for the capability of
extracting information those VMS to be interpreted by AV. The goal of this research was to build a
system to extract Regions of Interest (ROI) from an image and classify whether that ROI contains a VMS
or not. Specifically, this project is looking to build and label a database of images containing VMS,
implement a method to segment the images based on ROI likely containing VMS, evaluate various
features by which a VMS may be distinguished, and train a Machine learning model to accurately classify
a Variable message sign. The XGBoost model built off the metrics extracted from the generated
database showed a precision, recall, and F1 score of approximately 0.87 when classifying variable
message signs in a 10-fold cross validation scoring of the extracted features. An additional item of note,
most failures to classify VMS could also be seen as exceptionally poor instances by segmentation
method identifying accurate ROI, leading to an improper classification. This lends to the idea that the
machine learning model may perform better with an improved segmentation method, additional
metrics allowing for proper classification of VMS within improperly segmented images, or a model which
is decoupled from the segmentation method and can accurately identify VMS with imperfect
segmentations.

###Usage
Use the notebooks for easiest manipulation, no images are included in the repository; however, there is a 
notebook for extracting feature information to create your own dataset given a local database of images, 
trained models are included under './models' 

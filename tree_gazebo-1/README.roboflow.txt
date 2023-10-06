
tree_gazebo - v1 2023-10-02 10:00pm
==============================

This dataset was exported via roboflow.com on October 2, 2023 at 2:02 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 144 images.
Tree are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 51 percent of the image
* Random rotation of between -24 and +24 degrees
* Random shear of between -23° to +23° horizontally and -22° to +22° vertically
* Random brigthness adjustment of between -36 and +36 percent
* Random Gaussian blur of between 0 and 5.25 pixels

The following transformations were applied to the bounding boxes of each image:
* Random rotation of between -19 and +19 degrees
* Random shear of between -22° to +22° horizontally and -24° to +24° vertically



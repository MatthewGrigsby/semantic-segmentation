<p align="center">
<font size="5">
Auto-Labeler for Aerial Imagery <br>
</font>
Matthew Grigsby <br>
Senior Data Scientist, IBM <br>
matthew.grigsby@ibm.com
</font>
</p>


# Introduction
**Background:** \
Semantic segmentation of buildings in a high resolution aerial [image](data/images/base_cropped.png) (~6500 x ~12500 pixels at .6M resolution) that belong to distinct classes. This is an important task for many GIS applications, such as urban planning, disaster response, and environmental monitoring. The objective is to create a model that can be used to label new images with minimal human intervention.

**Goal of this repository**: \
Build initial model to prove out efficacy of segmentation architectures to complete this task using a single neural network. The resulting model should be able to segment the image with a reasonable degree of accuracy and speed. The model should also be able to generalize to new images.

**Future goals**: \
Incorporate in a semi-supervised learning system with minimal human in the loop interaction as needed. 
<br><br>

![image](data/readme_images/base_cropped.png)


# Methods 
My approach to solving the challenge of labeling arial imagery was to utilize cutting-edge models developed for semantic segmentation tasks. The first step was cleaning up and finishing the annotations we were given (found [here](data/archive/final_building_predictions_color.tif)). Once I was satisfied with the quality of the resulting annotated [image](data/annotations/) (the "ground truth"), I chopped it into overlapping tiles small enough for the model to ingest. Unfortunately, the original image was too large for ingesting all at once given software/hardware/time constraints. Furthermore, tiling the main image into overlapping pieces resulted in a set of images large enough to ease concerns regarding sample size when training a first draft model in proving out the efficacy of this approach. 

Now that I had our tiles created from the original image and mask, I decided to remove masks and their corresponding images which contained only the background class (segment 0 in grayscale). Note: a valid alternate structure would be to not include a background class (e.g. 0 would be the first building segment type). Once this was done, the tiles were ready for training and moved into [directories](data/data_for_training_and_testing_n1024) appropriate for my data loaders (TensorFlow's [ImageDataGenerator.flow_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory) class is particular about directory structure).

I then built out necessary classes for [data loaders](src/torch_scripts/torch_load_data.py), [model](src/torch_scripts/torch_model.py), and [specifications](src/torch_scripts/torch_main.py) for running our experiments. Initial runs were completed as a U-Net architecture with ResNet backbone and pretrained weights from [ImageNet](https://www.image-net.org/about.php). I ultimately decided upona a multi-scale attention netowrk (MA-Net) architecture with a SegFormer backbone and again pretrained weights from imagenet. Throughout the experimentation process, I tested varying combinations of [architectures](src/torch_scripts/torch_model.py) (U-Net, MA-Net, and FPN), [backbones](src/torch_scripts/torch_main.py) (ResNet(34, 50, 101, 152) and SegFormer mit_b1-3), [weights](src/torch_scripts/torch_main.py) (pretrained or random initialization), [batch sizes](src/torch_scripts/torch_main.py) (mostly dependent on model size and tile height/width), [loss functions](src/torch_scripts/torch_main.py) (Dice or Jaccard), and [image augmentations](src/torch_scripts/torch_load_data.py) (crop, flip, rotate, etc.). A record with most of my progress is saved [here](notes.txt). 
<br><br>

# Results 
[Full image](results/pretty_result_manet_mit_b3_imagenet_3b_6c_1024p_v2.png) intersection over union (IoU) score for the final model was 0.81. This is a great score considering the complexity of the image and the fact that the model was trained on a single, albiet very high resolution, aerial image. The model was able to generalize to new images, as shown [here](results/test_pretty_result_manet_mit_b3_imagenet_3b_6c_1024p_v2.png). The model was also able to label (patch into 1024x2014, augment, predict, and rebuild full image) the image in a relatively quick amount of time (~24sec). This is much faster compared to the City's original, poorer performing (in both speed and segmentation ability) solution that took approximately 36min to complete.

![Full image](data/readme_images/result.png)
<br>

# References
**Architectures:** \
U-Net: https://arxiv.org/pdf/1505.04597.pdf \
MA-Net: https://ieeexplore.ieee.org/abstract/document/9201310 \
FPN: https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Seferbekov_Feature_Pyramid_Network_CVPR_2018_paper.pdf

**Encoders/Backbones:** \
ResNet: https://arxiv.org/pdf/1512.03385.pdf \
SegFormer: https://arxiv.org/pdf/2105.15203.pdf 

**Segmentation models libraries** \
PyTorch: https://github.com/qubvel/segmentation_models.pytorch \
TensorFlow: https://github.com/qubvel/segmentation_models

<br>

# Steps to recreate segmentation model
1. Annotate the image, painting each building with corresponding segment color (files found [here](data/annotations))
2. Convert annotated image into grayscale: [notebook](src/create_cleaned_masks.ipynb)
3. Cut the main image into smaller tiles and setup file directories: [notebook](src/create_training_data.ipynb)
4. Specify model parameters such as epochs, batch size, loss function, etc.
     - [PyTorch version](src/torch_scripts/torch_main.py)
     - [TensorFlow version](src/tf_scripts/tf_main.py)
5. Create conda environment and install necessary packages
     - [PyTorch version](src/torch_scripts/requirements.txt)
     - [TensorFlow version](src/tf_scripts/requirements.txt)
6. Open a terminal, activate your environment, navigate to either the [PyTorch](src/torch_scripts/) or [TensorFlow](src/tf_scripts/) folders and run the corresponding main function. Training on GPU(s) is highly recommended.
7. View results once model finishes training:
     - View model results on single tile [here](src/view_model_results.ipynb)
     - View prediction on entire image [here](src/view_predictions_full_image.ipynb)
     - Model and epoch history are saved [here](models)
  
<br>

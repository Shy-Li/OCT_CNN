# OCT_CNN
A custormized ResNet model for OCT colorectal image classification.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Training the network](#training)
- [Testing the network](#testing)
- [Citation](#citation)

## Background
A residual neural network (ResNet)-based deep learning model manufactured and trained to perform automatic image processing and real-time diagnosis of the OCT images. 

## Install
The code was tested with Python 3.8.8 and TensorFlow 2.4.1.

Required packages: 

 - tensorflow-gpu 2.4.1
 - numpy 1.20.1
 - pandas 1.2.4
 - matplotlib 3.3.4
 - pillow 8.2.0
 
On Whitaker 160 ORIGIN GPU PC, use the enviroment OCT NN: 'conda activate OCTNN'.

## Training
Use `train.py` to train the ResNet. The training dataset and validation dataset contains both bechtop and catheter images. 
The pre-trained networks can be found in the folder 'models/models_train_on_both2'. 
## Testing
Use 'plot_prob.py' to plot prediction scores on all testing images.
Use 'test_result.py' to average over B-scan images, plot ROC, and calculate AUC.
 
## Citation
Luo H, Li S, Zeng Y, Cheema H, Otegbeye E, Ahmed S, Chapman WC Jr, Mutch M, Zhou C, Zhu Q. Human colorectal cancer tissue assessment using optical coherence tomography catheter and deep learning. J Biophotonics. 2022 Feb 11:e202100349. doi: 10.1002/jbio.202100349. Epub ahead of print. PMID: 35150067.

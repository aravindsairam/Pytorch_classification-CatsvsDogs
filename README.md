# Pytorch_classification-CatsvsDogs
Classification of Cats and Dogs with Pytorch

This is the code for Kaggle's Dogs vs. Cats Redux competition. 

Training dataset = 25,000 images and 
Testing dataset = 12,500 images

Network details:
- CNN Archiecture : Pretrained Resnet-34
- Learning rate : 0.0001 with Learning rate scheduler(Reduce on Plateau)
- Optimizer : Adam
- Batch size : 64
- epochs : 10
- augmentations : random horizontal flip, shift, scale and rotate. 
- Removes 51 wrong/incorrect images from train set which are not cats nor dogs (or both cats and dogs together).


Validation Loss : 0.095 (average of 5folds).


Achieved Public and Private score : 0.06599 Log Loss (mean ensemble of 5 models)
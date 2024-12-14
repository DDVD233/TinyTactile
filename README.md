# TinyTyper

## Overview

In the project, we develop a pressure-sensing textile interface that learns unique typing patterns through a convolutional neural network (CNN). By training the CNN on a calibration sequence of key presses, we enable next-character prediction tailored to individual users. We enhance this model with a language model for contextual guidance, improving accuracy, and apply fine-grained and channel pruning to optimize inference latency for real-time use. Our findings demonstrate the feasibility of lightweight, intelligent typing models for novel edge devices, paving the way for more adaptive and versatile input interfaces.

## Demo

[![demo-thumbnail](https://github.com/user-attachments/assets/68bd972e-c579-4d52-892d-cf8863842476)](https://vimeo.com/1039126496?share=copy#t=0)

## Set-up

### Prequisite
The typing interface is part of the WiReSens Toolkit. Instructions for programming the sensor can be found at [https://github.com/WiReSens-Toolkit](https://github.com/WiReSens-Toolkit). While the model can be trained without the sensor, the sensor is needed to perform typing actions.

### Training
The training script is `/src/train.py`, which consumes the datasets in `/recordings` to train the model. `train.py` takes in an argument, `--model`, to determine one of the 4 architectures explored in this project: `resnet`, `shallow_cnn`, `knn`, `svm`. The script also takes in an optional parameter, `--epochs`, for the number of epochs for the training phase.

### Pruning
Channel pruning and fine-grained pruning utilities can be found at `src/channel-prune.py` and `src/prune.py`, respectively. To produce a pruned model, update the script to reflect the name of the model you want to load and run the desired script.

## Poster Presentation

![Poster Presentation](https://github.com/user-attachments/assets/f75e2bc4-2f13-4b41-bc6c-0734ebcea28e)

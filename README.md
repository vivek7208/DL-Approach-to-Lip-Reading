# LipNet: Lip Reading Project

This project is based on the paper "LipNet: End-to-End Sentence-level Lipreading" by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, Nando de Freitas. 

## Paper Reference

The paper presents LipNet, an end-to-end trainable model for lip reading at the sentence level. The model makes use of spatiotemporal convolutions, a recurrent network, and the connectionist temporal classification loss. 

The model is trained to map a variable-length sequence of video frames to text, learning both spatiotemporal visual features and a sequence model simultaneously. On the GRID corpus, LipNet achieves 95.2% accuracy in sentence-level, overlapped speaker split task, outperforming experienced human lipreaders and the previous 86.4% word-level state-of-the-art accuracy.

The paper can be found at [this link](https://doi.org/10.48550/arXiv.1611.01599).

## Table of Contents

- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)

## Data Preparation

The first part of the notebook involves preparing the video data to be used for training the model. This involves several steps:

1. The function `load_video` is used to load a video from a given path. This function reads the video, frame by frame, converts the frames to grayscale, and normalizes the data. The frames are then returned as a list of tensors. The normalization is done by subtracting the mean of the frames and dividing by the standard deviation.

2. The function `load_alignments` is used to load the ground truth alignments for a video. The alignments are loaded from a text file where each line corresponds to a word spoken in the video. The function returns a list of tokens, each token corresponding to a word in the video.

3. The `load_data` function is then used to load both the video and its corresponding alignments. This function calls `load_video` and `load_alignments` and returns a tuple of the frames and alignments.

4. A vocabulary is created using all the unique characters that can be present in the output text. This includes all the lowercase English alphabets, digits from 1 to 9, and a few special characters. The vocabulary is then used to create two look-up layers, `char_to_num` and `num_to_char`, which convert characters to numbers and vice-versa, respectively.

5. The data is then prepared for training using the `tf.data.Dataset` API. All the video files are first listed and then shuffled. The `load_data` function is mapped onto the list of video files to load the data. The data is then batched and prefetched for efficient training.

## Model Architecture

The model used in this project is a sequential model, consisting of three main parts:

1. Conv3D Layers: The model starts with three Conv3D layers, each followed by a ReLU activation and a MaxPool3D layer. These layers are used to extract features from the video frames.

2. Bidirectional LSTM Layers: The output from the Conv3D layers is then passed through two Bidirectional LSTM layers. Each LSTM layer is followed by a Dropout layer for regularization.

3. Dense Layer: The final layer is a Dense layer with a softmax activation function. The number of units in this layer is equal to the size of the vocabulary plus one (for the blank character).

Here is the detailed structure of the model:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv3d (Conv3D)             (None, 75, 46, 140, 128)  3584      
                                                                 
 activation (Activation)     (None, 75, 46, 140, 128)  0         
                                                                 
 max_pooling3d (MaxPooling3D  (None, 75, 23, 70, 128)  0         
 )                                                               
                                                                 
 conv3d_1 (Conv3D)           (None, 75, 23, 70, 256)   884992    
                                                                 
 activation_1 (Activation)   (None, 75, 23, 70, 256)   0         
                                                                 
 max_pooling3d_1 (MaxPooling  (None, 75, 11, 35, 256)  0         
 3D)                                                             
                                                                 
 conv3d_2 (Conv3D)           (None, 75, 11, 35, 75)    518475    
                                                                 
 activation_2 (Activation)   (None, 75, 11, 35, 75)    0         
                                                                 
 max_pooling3d_2 (MaxPooling  (None, 75, 5, 17, 75)    0         
 3D)                                                             
                                                                 
 time_distributed (TimeDistr  (None, 75, 6375)         0         
 ibuted)                                                         
                                                                 
 bidirectional (Bidirectiona  (None, 75, 256)          6660096   
 l)                                                              
                                                                 
 dropout (Dropout)           (None, 75, 256)           0         
                                                                 
 bidirectional_1 (Bidirectio  (None, 75, 256)          394240   
 nal)                                                            
                                                                 
 dropout_1 (Dropout)         (None, 75, 256)           0         
                                                                 
 dense (Dense)               (None, 75, 41)            10537     
                                                                 
=================================================================
Total params: 8,471,924
Trainable params: 8,471,924
Non-trainable params: 0
```

## Training

The model is trained using the Adam optimizer with a learning rate of 0.0001. The learning rate is decayed exponentially after 30 epochs to help the model converge. The model is trained for a total of 100 epochs.

During training, the model's weights are saved after each epoch if the loss on the validation set improves. This is done using the ModelCheckpoint callback.

A custom callback, ProduceExample, is also used during training. This callback prints the model's predictions for a sample from the test set after each epoch. This helps in monitoring the model's learning progress.

## Inference

The trained model can be used for inference on new videos. The video is first loaded and preprocessed using the `load_data` function. The model's predictions are then obtained using the `model.predict` function. The predictions are sequences of numbers, which are converted back to text using the `num_to_char` look-up layer.

The model's predictions can be evaluated using the ground truth text for the videos. This can be done by comparing the model's predictions with the actual text spoken in the videos.

The model can also be evaluated on the test set. The test set is loaded in the same way as the training set. The model's performance on the test set gives an unbiased estimate of its ability to generalize to unseen data.

The project demonstrates the potential of deep learning for lip reading, following the ideas proposed in the referenced paper. The model could potentially be improved by using more data, tuning the hyperparameters, or using a more complex model architecture.

## Model Visualization

Placeholder for GIF showing the model in action:

![Model in Action](path_to_gif.gif)

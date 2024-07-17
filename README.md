# TripletFaceVerify
Deep Learning for Face Verification with Triplet Loss

Welcome to my Face Verification project using a Siamese network and triplet loss. This project aims to verify faces by comparing an anchor image with positive and negative images. The Siamese network, trained with triplet loss, ensures that the anchor and positive images (both of my own photos) are closer in the embedding space, while the anchor and negative images (randomly selected from the LFW dataset) are farther apart. This project demonstrates the effectiveness of using deep learning techniques for face verification tasks.

## Loss function 
The Loss function is L=max(0,∥f(x<sup>a</sup>)−f(x<sup>p</sup>)∥<sup>2</sup>−∥f(x<sup>a</sup>)−f(x<sup>n</sup>)∥<sup>2</sup>+α)
## Dataset 
The dataset for negative images can be  accesssed from https://www.kaggle.com/datasets/jessicali9530/lfw-dataset<br>
The positive and anchor images are custom images of myself. 

## Model over View 
1.Data Preparation:<br>
    Collect and preprocess images: Anchor and positive images are your own photos, while negative images are sourced from the LFW dataset.<br>
    Create triplets: Generate anchor-positive-negative image triplets for training.<br>
2.Model Architecture:<br>
    Siamese Network: Design a Siamese network with shared weights to learn embeddings for images.<br>
    Embedding Layer: Use a convolutional neural network (CNN) to extract feature embeddings from images.<br>
3.Loss Function:<br>
    Implement the triplet loss function to ensure that the anchor-positive distance is smaller than the anchor-negative distance by a margin.<br>
4.Training:<br>
    Train the network using the triplets, minimizing the triplet loss function to learn meaningful embeddings.<br>
5.Evaluation:<br>
    Test the model on new face pairs to verify its ability to distinguish between genuine and impostor faces.<br>
6.Deployment:<br>
    Use the trained model to perform face verification on new images, comparing embeddings to determine if they belong to the same person.<br>

## Traiend Model
The Tranined model file can be accessed from https://drive.google.com/file/d/1hlGaQ7jSR39OA4q3DzSHODPy4sn44DLX/view?usp=sharing

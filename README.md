# DeepLearning

Projects for Deep Learning specialiaztion on Coursera

* Art Generation - Add a style to a given image, by using pretrained VGG-19 model and tensorflow V1

* Character-Level Language Modeling - for two tasks
    * Generate a dinosaur name -> using RNN
    * Generate a Shakespeare-like writing -> using LSTM
TensorFlow V2 is used in this notebook

* Convolutional Neural Network Tutorial:
    * Function for the main building blocks of CNN (Forward propagation, Max Pool, Fully Connected and back propagation)
    * Using TensorFlow to build CNN to improve the hand sign classifier

* Date Translation - Using Attention model and LSTM the model takes any date format and outputs "YYYY-MM-DD" format

* Deep Neural Network - Building L-Layer Deep Neural Network and Improve the cat classifier

* Emojify - Predict the emoji for a given sentence
    * V1 - Using GloVe encoding by averaging the encoded vectors and softmax to find the output
    * V2 - Adding LSTM to consider the order of words not just the encoding vector

* Face Recognition - Recognize the person in image by doing the next steps:
    * Encode the image using pretrained inception network
    * Compare the encoded vector with stored vectors
    * Predict by using a given threshold for the distance between the vectors
 TensorFlowV1 is used

* Jazz Improvisation with LSTM - Generate Jazz music using LSTM and Keras

* Logisitc Regression - Implementing LR and Gradient Descent to understand how it's work and building a simple cat classifier with it.

* Optimization Algorithm - Comparision between different optimization algorithms; gradient descent, momentum and ADAM.

* RNN Step by Step - Building all components and units of RNN without frameworks
    * Forward and Back propagation for simple RNN and LSTM

* ResNet - Building ResNet blocks using Keras and use it to build a better hand signs detector

* Word Vectors - Using GloVe dataset and cosine similarity find the relation between words

* TensorFlow Tutorial - Building a neural network to detect hand signs (0-1-2-3-4-5) from image. In the original project, I used TF version 1, but lately, in order to migrate to tf version 2, I updated the code. Replacing version 1 functions with the newer version equivalents. Calls for sessions are removed and placeholders are no longer available.

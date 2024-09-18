<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Binary Classification of Brain Tumor Using CNN</h1>

<h2>Project Overview</h2>
<p>This project implements a Convolutional Neural Network (CNN) to classify brain tumor images into two categories: <strong>tumor</strong> and <strong>Healthy</strong>. The network is designed to extract hierarchical features from the images through multiple convolutional layers and uses a sigmoid activation function in the final layer to output probabilities for binary classification.</p>

<h2>Model Architecture</h2>
<ul>
  <li><strong>Convolutional Layers:</strong> Five convolutional layers with increasing filter sizes (64, 128, 256, 364, 684) for hierarchical feature extraction from the brain MRI images.</li>
  <li><strong>Activation Function:</strong> ReLU (Rectified Linear Unit) activation is used for all hidden layers, providing non-linearity to the network.</li>
  <li><strong>MaxPooling Layers:</strong> After each convolutional layer, max pooling reduces the spatial dimensions of the image, helping the model generalize better to unseen data.</li>
  <li><strong>Dropout Layers:</strong> Dropout is applied after the second, third, and fourth convolutional layers to reduce overfitting and enhance generalization.</li>
  <li><strong>Fully Connected Layers:</strong> Two dense layers are used before the final output layer to consolidate the features extracted from the convolutional layers.</li>
  <li><strong>Output Layer:</strong> A single node with sigmoid activation is used to output the probability of the image being classified as a tumor or non-tumor.</li>
</ul>

<h2>Training</h2>
<ul>
  <li>The model is compiled with the <strong>RMSprop</strong> optimizer and uses <strong>binary cross-entropy</strong> as the loss function, which is well-suited for binary classification tasks.</li>
  <li>The model is trained for 50 epochs with a batch size of 4, and validation is performed on 10% of the training data.</li>
</ul>

<h2>Results</h2>
<p>The CNN model achieved an accuracy of <strong>93.75%</strong> on the binary classification task. This demonstrates that the model is effective in distinguishing between tumor and non-tumor images in the dataset.</p>

<h2>Conclusion</h2>
<p>The model's accuracy of 93.75% indicates that it is capable of reliably classifying brain tumor images. The architecture, including convolutional layers for feature extraction and dropout layers for regularization, helped the model generalize well to the validation data. Future improvements could involve further tuning of the hyperparameters, data augmentation, or testing on more complex datasets.</p>

<h2>Installation</h2>
<ol>
  <li>Clone the repository:</li>
  <pre><code>git clone https://github.com/Mukund604/Binary-Classification-Of-Brain-Tumors-using-CNN</code></pre>
  <li>Navigate to the project directory:</li>
  <pre><code>cd brain-tumor-classification</code></pre>
  <li>Install the required dependencies:</li>
  <pre><code>pip install -r requirements.txt</code></pre>
  <li>Run the training script:</li>
  <pre><code>python train_model.py</code></pre>
</ol>

<h2>Usage</h2>
<p>After training, you can use the trained model to classify new brain MRI images as either tumor or non-tumor. Simply pass the image through the model and it will output a probability score.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>

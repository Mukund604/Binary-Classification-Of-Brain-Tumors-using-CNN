# Binary Classification of Brain Tumors Using CNN

A deep Convolutional Neural Network (CNN) built with TensorFlow/Keras for binary classification of brain MRI images into **Tumor** and **Healthy** categories. Achieves **93.75% accuracy** on the test set.

## Dataset

- **Source:** [Brain Cancer Detection MRI Images](https://www.kaggle.com/datasets/ismailpromus/brain-cancer-detection-mri-images) on Kaggle
- **Size:** 800 grayscale MRI images (resized to 100x100)
- **Classes:** Tumor (1) and Healthy (0)
- **Split:** 80% train, 20% test

## Model Architecture

- **5 Convolutional layers** with increasing filters (64, 128, 256, 364, 684) and ReLU activation
- **MaxPooling** after each convolutional layer
- **Dropout (0.2)** after the 2nd, 3rd, 4th, and 5th convolutional layers
- **2 Dense layers** (128 and 64 units) with ReLU activation
- **Output layer** with sigmoid activation for binary classification

## Training Configuration

- **Optimizer:** RMSprop
- **Loss:** Binary cross-entropy
- **Epochs:** 50
- **Batch size:** 4
- **Validation split:** 10% of training data

## Results

The model achieved **93.75% accuracy** on the test set, demonstrating effective feature extraction and generalization for brain tumor classification.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Mukund604/Binary-Classification-Of-Brain-Tumors-using-CNN.git
   cd Binary-Classification-Of-Brain-Tumors-using-CNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the [dataset from Kaggle](https://www.kaggle.com/datasets/ismailpromus/brain-cancer-detection-mri-images) and update the dataset path in the notebook.

4. Open and run the notebook:
   ```bash
   jupyter notebook binary-classification-of-brain-tumors-using-cnn.ipynb
   ```

## Dependencies

- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Project Structure

```
.
├── binary-classification-of-brain-tumors-using-cnn.ipynb   # Main notebook
├── archive.zip                                              # Dataset archive
├── requirements.txt                                         # Python dependencies
└── README.md
```

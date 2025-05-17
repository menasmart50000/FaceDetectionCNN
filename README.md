<h1>CNN Face Recognition Model</h1>

This project implements a Convolutional Neural Network (CNN) for face recognition using the Labeled Faces in the Wild (LFW) dataset. The model is built with TensorFlow/Keras, preprocesses images, trains the CNN, and evaluates performance with metrics and visualizations.

<h2>Features</h2>

Loads and preprocesses the LFW dataset.

Applies image preprocessing: reshaping, normalization, and data augmentation.

Builds a CNN model with convolutional, pooling, and dense layers.

Trains the model with a low learning rate and data augmentation.

Evaluates performance using accuracy, precision, recall, F1-score, confusion matrix, and training history plots.

Prints preprocessing step outputs for transparency.
Prerequisites

Ensure you have Python 3.8+ installed. The required libraries are:





tensorflow (2.10+)



scikit-learn



numpy



matplotlib



seaborn



opencv-python

Install dependencies using:

pip install tensorflow scikit-learn numpy matplotlib seaborn opencv-python

Installation





Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name



Install the required libraries (see Prerequisites).



Ensure you have an active internet connection, as the LFW dataset is downloaded automatically by scikit-learn.

Usage





Place the main script (face_recognition_cnn.py) in your project directory.



Run the script:

python face_recognition_cnn.py



The script will:





Download and preprocess the LFW dataset.



Train the CNN model for 20 epochs.



Print preprocessing details (shapes, pixel ranges, augmentation settings).



Output performance metrics (accuracy, precision, recall, F1-score).



Save two plots in the current directory:





confusion_matrix.png: Confusion matrix heatmap.



training_history.png: Training and validation accuracy/loss plots.

<h2>File Structure</h2>

your-repo-name/
├── face_recognition_cnn.py   # Main script for the CNN face recognition model
├── README.md                 # Project documentation
├── confusion_matrix.png      # (Generated) Confusion matrix plot
└── training_history.png      # (Generated) Training history plots

Code Overview

The main script (face_recognition_cnn.py) performs the following steps:





Imports Libraries: TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Seaborn, and OpenCV.



Loads LFW Dataset: Uses fetch_lfw_people to load face images (minimum 70 images per person).



<h2>Preprocesses Images:</h2>





Reshapes images to include a channel dimension.



Normalizes pixel values to [0, 1].



Configures data augmentation (rotation, shifts, flips).



Splits data into 80% training and 20% testing sets.



Prints preprocessing details (shapes, pixel ranges, augmentation settings, split sizes).



Builds CNN Model: Defines a Sequential model with three convolutional layers, max-pooling, dropout, and dense layers.



<h2>Trains Model:</h2> 

Uses Adam optimizer (learning rate 0.0001), trains for 20 epochs with augmentation.



<h2>Evaluates Performance:</h2>

Computes metrics and saves confusion matrix and training history plots.

Output





Console Output:

Final metrics: accuracy, precision, recall, F1-score.
Saved Files:
confusion_matrix.png: Visualizes true vs. predicted labels.

training_history.png: Plots training/validation accuracy and loss.

Example Preprocessing Output

After reshaping:
  Shape: (1288, 50, 37, 1)
  Data type: float64

After normalization:
  Min pixel value: 0.0000
  Max pixel value: 1.0000
  Data type: float32

Data augmentation configuration:
  Rotation range: 10 degrees
  Width shift range: 10%
  Height shift range: 10%
  Horizontal flip: Enabled

<h1>
  After dataset splitting:
</h1> 
  Training set shape: (1030, 50, 37, 1), Labels shape: (1030,)
  Testing set shape: (258, 50, 37, 1), Labels shape: (258,)

Notes





The LFW dataset is downloaded automatically by scikit-learn on the first run.



The model trains for 20 epochs, which may take several minutes depending on your hardware.



The plots are saved in the current working directory. Check your directory with:

import os
print(os.getcwd())



To modify the save location, update the plt.savefig() calls in face_recognition_cnn.py.

Contributing

Contributions are welcome! Please:





Fork the repository.



Create a feature branch (git checkout -b feature-name).



Commit changes (git commit -m 'Add feature').



Push to the branch (git push origin feature-name).



Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For questions or issues, please open an issue on GitHub or contact your-email@example.com.

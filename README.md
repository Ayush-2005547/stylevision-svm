# Fashion MNIST Classification using Support Vector Machine (SVM)

This project demonstrates the use of Support Vector Machines (SVM) to classify fashion items in the Fashion MNIST dataset. We apply image preprocessing techniques, extract features using Histogram of Oriented Gradients (HOG), and use an SVM with RBF kernel to train a robust classifier.

---

## Project Overview

- **Dataset:** Fashion MNIST (60,000 training, 10,000 testing grayscale images of 28x28 pixels)
- **Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Model:** Support Vector Machine (SVM) with RBF kernel
- **Feature Extraction:** Histogram of Oriented Gradients (HOG)

---

## Objectives

- Explore image classification using SVM
- Extract meaningful features from images using HOG
- Evaluate model performance using standard metrics
- Demonstrate SVM usage in real-world image classification tasks

---

## Tools & Libraries

- Python
- Scikit-learn
- NumPy
- Matplotlib
- skimage (for HOG feature extraction)

---

## Implementation Steps

1. **Load Dataset:** Used `fetch_openml` to load Fashion MNIST.
2. **Preprocessing:** Normalized pixel values; converted labels to integers.
3. **HOG Feature Extraction:** Extracted texture and edge features from each image.
4. **Train-Test Split:** Split the dataset into training and testing sets.
5. **Feature Scaling:** Standardized features for optimal SVM performance.
6. **Model Training:** Trained an SVM classifier with RBF kernel.
7. **Evaluation:** Used accuracy score, confusion matrix, and classification report.
8. **Visualization:** Displayed a few predictions vs actual labels.

---

## Results

- **Accuracy Achieved:** ~77-80% (may vary based on parameters)
- **Model Strengths:** Performs well with non-linear data using RBF kernel
- **Challenges Addressed:** Feature extraction for better image-based learning

---

## Conclusion

This project shows that SVMs, when combined with effective feature extraction like HOG, can be powerful tools for image classification tasks. The Fashion MNIST dataset provided a meaningful challenge and demonstrated the practical application of machine learning in fashion-related tasks.

---

## Future Enhancements

- Deploy the model using Streamlit or Flask for real-time predictions
- Implement hyperparameter tuning using GridSearchCV
- Experiment with alternative kernels or dimensionality reduction (e.g., PCA)


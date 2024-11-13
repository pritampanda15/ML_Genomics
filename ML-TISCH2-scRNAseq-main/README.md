# ML-TISCH2-scRNAseq
A Machine-Learning regression models implemented in Python, including Ridge, Random Forest, Gradient Boosting, SVR, and a neural network using Keras for scRNAseq datasets from TISCH2 analyzed using Seurat package. 

## Overview
This project demonstrates the use of different regression techniques to predict continuous outcomes. The models included are:
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- Neural Network (Keras)

## Installation
To run the models, you need to have Python installed along with the necessary libraries. You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
### Additional Files
1. **requirements.txt**: List of dependencies for your project.
2. **scripts/**: Directory to store your model training and evaluation scripts.
3. **data/**: Directory to store your dataset (if applicable).
4. **results/**: Directory to store the results and saved models.

### Example requirements.txt
```plaintext
tensorflow
scikeras
scikit-learn
joblib
numpy
pandas
matplotlib
```

### Usage

    Data Preparation: Ensure your data is properly formatted and preprocessed.
    Training Models: Use the provided scripts to train the different regression models.
    Evaluating Models: Evaluate the performance of each model using appropriate metrics.


### Models

    Ridge Regression: Regularized linear regression to prevent overfitting.
    Random Forest Regressor: Ensemble method that uses multiple decision trees.
    Gradient Boosting Regressor: Builds models sequentially to minimize error.
    Support Vector Regressor (SVR): Uses support vector machines for regression tasks.
    Neural Network (Keras): Deep learning model implemented using Keras.

### Results

The performance of each model is evaluated using mean squared error (MSE) and other relevant metrics. Detailed results can be found in the results directory.

### Visualization
The architecture of the neural network model is saved as an image

### Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss improvements or new features.

### License

This project is licensed under the MIT License.

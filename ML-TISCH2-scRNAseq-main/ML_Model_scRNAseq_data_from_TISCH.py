#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, SGDClassifier, LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, PassiveAggressiveClassifier, Perceptron
from sklearn.svm import SVR, SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pydot
import graphviz
import joblib


# In[56]:


# Load the dataset
data = pd.read_csv('CRC_datasets.tsv',sep='\t')


# In[57]:


data


# In[58]:


# Select features and target
features = data[['Cluster', 'Celltype (malignancy)', 'Celltype (major-lineage)', 'Celltype (minor-lineage)', 'Gene', 'Percentage (%)', 'Adjusted p-value']]
target = data['log2FC']


# In[59]:


target


# In[60]:


# Handle categorical features
categorical_features = ['Cluster', 'Celltype (malignancy)', 'Celltype (major-lineage)', 'Celltype (minor-lineage)', 'Gene']
numeric_features = ['Percentage (%)', 'Adjusted p-value']


# In[61]:


# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# In[62]:


# Fit the preprocessor on the entire dataset
preprocessor.fit(features)


# In[63]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[64]:


# Preprocess the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


# In[65]:


# Save the preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')


# In[50]:


# Define all classification and regression models models and hyperparameters
#models = {
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'SVR': SVR(),
    'LinearSVC': LinearSVC(),
    'SGDClassifier': SGDClassifier(),
    'MLPClassifier': MLPClassifier(),
    'Perceptron': Perceptron(),
    'LogisticRegression': LogisticRegression(),
    'LogisticRegressionCV': LogisticRegressionCV(),
    'SVC': SVC(),
    'CalibratedClassifierCV': CalibratedClassifierCV(),
    'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
    'LabelPropagation': LabelPropagation(),
    'LabelSpreading': LabelSpreading(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier(),
    'RidgeClassifierCV': RidgeClassifierCV(),
    'RidgeClassifier': RidgeClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'BernoulliNB': BernoulliNB(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'GaussianNB': GaussianNB(),
    'NuSVC': NuSVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'NearestCentroid': NearestCentroid(),
    'ExtraTreeClassifier': ExtraTreeClassifier()
}


# In[51]:


#param_grids = {
    'Ridge': {
        'alpha': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'SVR': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    'LinearSVC': {
        'C': [0.1, 1, 10],
        'max_iter': [1000, 5000, 10000]
    },
    'SGDClassifier': {
        'loss': ['hinge', 'log', 'modified_huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000, 3000]
    },
    'MLPClassifier': {
        'hidden_layer_sizes': [(50,), (100,), (150,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    },
    'Perceptron': {
        'penalty': [None, 'l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000, 3000]
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear', 'saga']
    },
    'LogisticRegressionCV': {
        'Cs': [1, 10, 100],
        'cv': [5, 10]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    'CalibratedClassifierCV': {
        'method': ['sigmoid', 'isotonic'],
        'cv': [3, 5, 10]
    },
    'PassiveAggressiveClassifier': {
        'C': [0.1, 1, 10],
        'max_iter': [1000, 2000, 3000]
    },
    'LabelPropagation': {
        'kernel': ['knn', 'rbf'],
        'n_neighbors': [3, 5, 7]
    },
    'LabelSpreading': {
        'kernel': ['knn', 'rbf'],
        'n_neighbors': [3, 5, 7]
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'QuadraticDiscriminantAnalysis': {
        'reg_param': [0.0, 0.1, 0.5, 1.0]
    },
    'HistGradientBoostingClassifier': {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_iter': [100, 200, 300]
    },
    'RidgeClassifierCV': {
        'alphas': [0.1, 1, 10],
        'cv': [3, 5, 10]
    },
    'RidgeClassifier': {
        'alpha': [0.1, 1, 10]
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'ExtraTreesClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'BaggingClassifier': {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0]
    },
    'BernoulliNB': {
        'alpha': [0.1, 0.5, 1.0]
    },
    'LinearDiscriminantAnalysis': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5, 1.0]
    },
    'GaussianNB': {
        'var_smoothing': [1e-09, 1e-08, 1e-07]
    },
    'NuSVC': {
        'nu': [0.1, 0.5, 1.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    'DecisionTreeClassifier': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'NearestCentroid': {
        'metric': ['euclidean', 'manhattan']
    },
    'ExtraTreeClassifier': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}


# In[83]:


# for this datasets we will be using regression models 
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    #'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    #'HistGradientBoosting': HistGradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    #'ExtraTrees': ExtraTreesRegressor(),
    'SVR': SVR(),
    'KNeighbors': KNeighborsRegressor(),
    'DecisionTree': DecisionTreeRegressor()
}


# In[84]:


# for this datasets we will be using regression models 
param_grids = {
    'LinearRegression': {},  # No hyperparameters to tune for LinearRegression
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'Lasso': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'ElasticNet': {
        'alpha': [0.01, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.5, 0.9]
    },
    #'RandomForest': {
        #'n_estimators': [100, 200, 300],
        #'max_depth': [10, 20, 30],
        #'min_samples_split': [2, 5, 10],
        #'min_samples_leaf': [1, 2, 4]
   # },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    #'HistGradientBoosting': {
        #'learning_rate': [0.01, 0.1, 0.2],
        #'max_iter': [100, 200, 300],
        #'max_depth': [3, 5, 7]
    #},
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    #'ExtraTrees': {
        #'n_estimators': [100, 200, 300],
        #'max_depth': [10, 20, 30],
        #'min_samples_split': [2, 5, 10],
        #'min_samples_leaf': [1, 2, 4]
    #},
    'SVR': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}


# In[85]:


# Train and tune models
best_models = {}
best_params = {}
best_scores_mse = {}
best_scores_accuracy = {}


# In[ ]:


for model_name, model in models.items():
    print(f"Training {model_name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    grid_search = GridSearchCV(pipeline, {'model__' + k: v for k, v in param_grids[model_name].items()}, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_models[model_name] = grid_search.best_estimator_
    best_params[model_name] = grid_search.best_params_
    best_scores_mse[model_name] = -grid_search.best_score_
    
    # Calculate accuracy on the training set
    y_pred = best_models[model_name].predict(X_train)
    if 'Classifier' in model_name or 'SVC' in model_name or 'NuSVC' in model_name or 'DiscriminantAnalysis' in model_name or 'BernoulliNB' in model_name or 'GaussianNB' in model_name or 'NearestCentroid' in model_name or 'ExtraTreeClassifier' in model_name:
        best_scores_accuracy[model_name] = accuracy_score(y_train, y_pred)
    else:
        best_scores_accuracy[model_name] = None

    print(f"Best params for {model_name}: {best_params[model_name]}")
    print(f"Best MSE for {model_name}: {best_scores_mse[model_name]}")
    print(f"Best accuracy for {model_name}: {best_scores_accuracy[model_name]}")

# Print the best models and their parameters
for model_name in best_models:
    print(f"Best {model_name}: {best_models[model_name]}")
    print(f"Best Parameters: {best_params[model_name]}")
    print(f"Best MSE: {best_scores_mse[model_name]}")
    print(f"Best Accuracy: {best_scores_accuracy[model_name]}")


# In[ ]:





# In[ ]:


# Evaluate models on the test set
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} Performance on Test Set:")
    print(f"  MSE: {mse}")
    print(f"  MAE: {mae}")
    print(f"  R^2: {r2}")
    print()


# In[ ]:


# Define a simple neural network model
def create_nn_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_transformed.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[ ]:


# Save the best model (example: Ridge)
nn_model.fit(X_train_transformed, y_train)


# In[ ]:


# Physics informed neural network
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming features and target are already defined
# Split the data
#X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Preprocess the data
# Assuming preprocessor is already defined
#X_train_transformed = preprocessor.fit_transform(X_train)
#X_test_transformed = preprocessor.transform(X_test)

# Define the neural network model using Keras Model subclassing
class PINN(Model):
    def __init__(self, input_dim):
        super(PINN, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.out = Dense(1, activation='linear')

    def call(self, X):
        x = self.dense1(X)
        x = self.dense2(x)
        return self.out(x)

    def pinn_loss(self, X, u_true, f_true):
        with tf.GradientTape() as tape:
            tape.watch(X)
            u_pred = self(X)
            u_x = tape.gradient(u_pred, X)
        f_pred = -tf.reduce_sum(u_x, axis=1, keepdims=True)  # Example: Simplified Poisson residual
        mse_u = tf.reduce_mean(tf.square(u_true - u_pred))
        mse_f = tf.reduce_mean(tf.square(f_true - f_pred))
        return mse_u + mse_f

    def train_step(self, data):
        X, u_true, f_true = data
        with tf.GradientTape() as tape:
            loss = self.pinn_loss(X, u_true, f_true)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}

# Sample f_train (physics-based target) assuming it's predefined or derived
f_train = np.ones_like(y_train, dtype=np.float32)  # Example physics constraint

# Initialize and compile the model
input_dim = X_train_transformed.shape[1]
model = PINN(input_dim)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Train the model using the fit method
model.fit(x=(X_train_transformed, y_train, f_train), y=y_train, epochs=1000, batch_size=32)

# Test the model
u_test_pred = model.predict(X_test_transformed)
print(u_test_pred)


# In[ ]:


joblib.dump(best_models['Ridge'], 'best_ridge_model_pipeline.pkl')


# In[ ]:


best_ridge_model_pipeline = joblib.load('best_ridge_model_pipeline.pkl')


# In[ ]:


y_pred = best_ridge_model_pipeline.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


# Create a DataFrame to compare predictions with actual values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Display the comparison
print(comparison_df.head())


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score

# Load the saved pipeline
loaded_pipeline = joblib.load('best_ridge_model_pipeline.pkl')

# Make predictions
y_pred = loaded_pipeline.predict(X_test)

# Evaluate the predictions
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


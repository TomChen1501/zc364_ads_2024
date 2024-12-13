# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_explanatory_variables_student_percentage(data, features):
    X = np.vstack([data[f'{feature}_weight'] for _, feature in features] + [data['school_weight'], data['population density']])
    X = X.T
    X = sm.add_constant(X, has_constant='add')
    return X

def get_response_variable_student_percentage(data):
    y = data['Normalized Observation']
    return y

def get_frequency_weights(data):
    return data['Total Observations']

def GLM_train(X, y, freq_weights=None):
    model = sm.GLM(y, X, family=sm.families.Gaussian(), freq_weights = freq_weights)
    return model.fit()

def GLM_predict(model, X):
    return model.predict(X)

def plot_correration_matrix(correlation_features, result_gdf):
    correlation_matrix = result_gdf[correlation_features].corr()

    # Plot a heatmap
    plt.figure(figsize=(21, 14))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Feature Correlation Heatmap with Target Variable (y)')
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(result_gdf, explanatory_variable,target_variable, predicted_variable):
    plt.scatter(result_gdf[explanatory_variable], result_gdf[target_variable], label='Actual', alpha=0.5, s=0.5)
    plt.scatter(result_gdf[explanatory_variable], result_gdf[predicted_variable], label='Predicted', color='red', alpha=0.5, s=0.4)
    plt.xlabel(explanatory_variable)
    plt.ylabel(target_variable)
    plt.legend()
    plt.title('Actual vs Predicted')
    plt.show()

def get_explanatory_variables_dependent_children(data, features):
    X = np.vstack([data[f'{feature}_weight'] for _, feature in features])
    X = X.T
    X = sm.add_constant(X)
    return X

def get_response_variable_dependent_children(data):
    y = data['Normalized dependent children']
    return y

def get_frequency_weights_dependent_children(data):
    return data[' Total number of dependent children: 2021 ']


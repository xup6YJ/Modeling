# Modeling
This is a modeling and evaluation example for a binary class structure data classification, modeling techniques including Machine Learning and Deep Learning. Bootstrap modeling and Result Plotting functions are included. 

# Universal Function
- ["Evaluation.py"](https://github.com/xup6YJ/Modeling/blob/main/Code/Evaluation.py) for outputting prediction result including Confusion Matrix, ROCurve, Sensitivity, Specificity, PPV, NPV, F1-Score(used by all of the modeling source code). Plotting confusion matrix on heap map as example below(Function 'plot_confusion').

<p align="center">
  <img src="Example Image/Confusion Matrix.jpg">
</p>

- To build different kinds of model in basic, including Random Forest, Logistic Regression, Support Vector Machine, DNN, RNN by using ["Model.py"](https://github.com/xup6YJ/Modeling/blob/main/Code/Model.py).

# Multi-model Comparison in ROCurve and Bootstrapping
Step 1.["DataPreprocessing.py"](https://github.com/xup6YJ/Modeling/blob/main/Code/DataPreprocessing.py)

Basic data feature engineering and spliting data into Train/Test dataset.

Step 2. ["Model_prediction.py"](https://github.com/xup6YJ/Modeling/blob/main/Code/Model_prediction.py)

Different model training and compare the proformance using ROCurve (Example figure as below).

<p align="center">
  <img src="Example Image/ROC.jpg">
</p>

Step 3. ["Model_bootstrapping.py"](https://github.com/xup6YJ/Modeling/blob/main/Code/Model_bootstrapping.py)

For more convincing result to compare those models, we perform bootstrapping and do ANOVA test to observe the model performance.

# Model Interpretation
Utilize ["Feature_importance.py"](https://github.com/xup6YJ/Modeling/blob/main/Code/Feature_importance.py) to know the importance of each features while training Deep Learning model.

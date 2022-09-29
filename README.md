# Overview

 Supporting information for *A review of supervised learning methods for classifying animal behavioral states from environmental  features* by Bergen et al.
 
 Subset of 10,000 GPS points and tutorial for implementing XGBoost to classify polytomous animal behavior from environmental features. 

* `eagle_subset.csv`: Subset of 10,000 GPS points with variables `risk_class` (3-level risk classification) and predictor features
* `xgboost_tutorial.Rmd`: R Markdown source file detailing implementation of XGBoost, specifically:
  - Model training
  - Model fitting
  - 5-fold CV and model assessment
  - Feature importance with SHAP values
  - ICE/PDP plots and SHAP plots
* xgboost_tutorial.docx: knitted Word version of tutorial.
* The other `.R` files provide code for searching a parameter grid, carrying out 5-fold cross-validation, and model assessment (pairwise area under ROC curve, confusion matrices, and by-class assessment metrics) for the other supervised learning methods:
  - `knn_train.R`: weighted k-nearest neighbors
  - `nnet_train.R`: neural nets
  - `randomforest_train.R`: random forests


# Warfarin_dose-Prediction
This project aims to use Neural Networks to predict warfarin dose for target INR value base on multiple features.  
  
The data used is modified from the source below.

**Original Dataset Source**  

International Warfarin Pharmacogenetics Consortium (IWPC)

Data from Estimation of the warfarin dose with clinical and pharmacogenetic data (PMID:19228618):
https://api.pharmgkb.org/v1/download/submission/553247439

**Reference**  

Truda, G., & Marais, P. (2021). Evaluating warfarin dosing models on multiple datasets with a novel software framework and evolutionary optimisation. Journal of Biomedical Informatics, 113, 103634. https://doi.org/10.1016/j.jbi.2020.103634
  
---
# Feature Importance Analysis before building Neural Network  
Using 9 machine learning models to find out best features for Neural Network:  
1. Decision Tree Regressor
2. Random Forest Regressor
3. Extra Trees Regressor
4. XGBoost Regressor
5. Gradient Boosting Regressor
6. Linear Regressor
7. Ridge Regressor
8. Lasso Regressor
9. ElasticNet Regressor

Result see `feature_importance.ipynb`.

---
# To Run
1. Run command ```pip install -r requirements.txt```
2. Go to UI directory
3. Run ```main.py```
  
---
# Old Version Directory  
Use regression model for warfarin dose prediction.  
Using Linear regression model, Poly linear regression model, and Random forest regression.  
  
## Models
The default model used is  Random Forest Regression.

To change the model, you need to replace the model path in ```main.py``` line 45 (```with open(model_path, 'rb') as filo:```).

Modedls are in models directory:
1. RFR.pkl - Random Forest Regression
2. lin.pkl - Linear Regression
3. poly.pkl - Polynomial Linear Regression
  
## To Run Old Version:
1. Go to the top level directory of old_version directory  
2. Run command ```pip install -r requirements.txt```
3. Go to UI directory
4. Run ```main.py```

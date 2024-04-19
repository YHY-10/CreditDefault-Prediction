# CreditDefault-Prediction
# 0. DataSource Description
  The data comes from the data set shared by HSBC Guest for the second time. The theme is to predict the loan default of a company based on its financial, internal behavior and other data.
# 1. DataAnalysis and Processing
## 1.1 Category Analysis via DataDictionary_V1.xlsx(1 Object and 104 int64/float64)
  There are three main modules: Bureau, Internal Behavior, and Financial, each module has more than 30 features. 
  There are 5 Segment features: Site, Age_of_Company_in_Month, Industry_Manufacturing_Flag, Industry_Wholesale_Trade_Flag, Industry_Services_Flag.
## 1.2 NAN Analysis via Transformed_data_v1.0_final.csv
  There are 75/105 features with missing values. Through the histogram and data description of DataDictionary_V1.xlsx below, we know that many of the features are limited to a certain segment.
## 1.3 Sample Imbalance Analysis and data split 8:2 via y Label
  I divided the training set and test set according to the ratio of 8:2 through label, and through the histogram below, we can see that the samples are extremely unbalanced.
## 1.4 Data Processing
Based on the above data analysis, we can perform the following data processing work
* **Drop CustomerID**: Since Drop CustomerID has no actual meaning, it is easy to add noise to model training, and the tree model will produce many split points. we will delete it
* **Nan filling with mean value**: We use the mean filling method.
* **Upsampling**: We use RandomOverSampler to upsample samples with label 1, which alleviates the sample imbalance problem to a certain extent.
* **'Site' One-hot Encoding**: Since the Site variable is an Object type feature, it contains 3 categories. It contains additional information, which we one-hot encode.
* **Segment Integration**: First Convert the Age_of_Company_in_Month segment feature into a categorical feature. Second, the values of the five segment features are synthesized to obtain a new segment feature "Integrated_segments", which is used to uniquely identify which segment a sample belongs to.
# 2. BenchMark Model
## 2.1 Logistic Model

## 2.2 LightGBM Model

| BenchMark Model\Evaluation index | AUC_Validation | AUC_Test    | Accuracy    | Recall      | Precision   | F1-score    |
|----------------------------------|----------------|-------------|-------------|-------------|-------------|-------------|
| Logistic Regression              | 0.724168589    | 0.679795904 | 0.73584242  | 0.524271845 | 0.071428571 | 0.12572759  |
| LightGBM                         | 0.997240439    | 0.771327333 | 0.931058741 | 0.300970874 | 0.2         | 0.240310078 |

## 2.3 LGB parameter Optimization via Optuna(An open-source framework for automated hyperparameter optimization)
I used optuna to search and adjust parameters such as 'num_leaves', 'learning_rate', 'max_depth', and 'n_estimators', and experimented 100 times.  
The optimal parameters obtained are as follows: Best params: {'num_leaves': 151, 'learning_rate': 0.04953824530034172, 'max_depth': 9, 'n_estimators': 713}  
The parameters and indicators are as follows:  
| BenchMark Model\Evaluation index | AUC_Validation | AUC_Test    | Accuracy    | Recall      | Precision   | F1-score    |
|----------------------------------|----------------|-------------|-------------|-------------|-------------|-------------|
| LightGBM_Optuna                  | 0.999805349    | 0.730986464 | 0.966232853 | 0.145631068 | 0.652173913 | 0.238095238 |
## 2.4 Feature Importance Analysis
The last 10% of features have little effect on model training and can be considered to be deleted directly in the future.  
# 3. New Model Structure
## 3.1 Structure Introduction
## 3.2 LGB + Logistic
## 3.3 XGB + Logistic
## 3.4 CatBoost + Logistic


# Phased Conclusions
# 4. Use AutoEncoder to get more features

## 4.1 AutoEncoder Structure
## 4.2 How I use it 
## 4.3 Traing Loss vs Validation Loss
## 4.3 Train LGB Model with more features

# 5. Conclusion


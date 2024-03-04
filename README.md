# credit-risk-classification
Module 20 Challenge

## Overview of the Analysis

In this analysis we attempted to build a model that would predict whether loans would be healthy or high-risk (likely to default). You can check out the testing process in the file in the main repo labeled `credit_risk_classification.ipynb`.

In order to build our models, we used lending data that be found in the Resources folder of this repo, labled `lending_data.csv` Risk factors being evaluated in this model included:
* Loan size
* Interest rate of the loan
* Income of the borrower
* The debt to income ratio fo the borrower
* Number of open accounts of the borrower
* Number of derogatory marks of the borrower
* The total debt of the borrower

### Analysis Steps
1. We divided our loan data into our target variable (`loan_status`) and the features (listed above). 
    - Upon further inspection of our target variables, we found that there were 30x more records on healthy loans than on high-risk loans. As result, we decided to run logisitic regressions on both the original data, as well as with balanced samples of that data. 
2. We ran a logistic regression on the first model with imbalanced data and evaluated the results.
3. Using `RandomOverSampler` from imbalanced-learn, we resampled our data to create a balanced number of healthy and high-risk loans. We then ran a logistic regression using this new re-sampled data and evaluated the results.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1(imbalanced data):
  * Balanced Accuracy Score: 0.9520479254722232
  * Healthy Loan Precision Score: 1.00
  * High-Risk Loan Precision Score: 0.85
  * Healthy Loan Recall Score: 0.99
  * High-Risk Loan Recall Score: 0.91

* Machine Learning Model 2 (balanced data):
  * Balanced Accuracy Score: 0.9936781215845847
  * Healthy Loan Precision Score: 1.00
  * High-Risk Loan Precision Score: 0.84
  * Healthy Loan Recall Score: 0.99
  * High-Risk Loan Recall Score: 0.99

## Summary
Machine Learning Model 2 appears to be a better predictor of the data. This makes sense as this Logistic Regression had more differentiated data in order to contrast the difference between healthy and high-risk loans. This is exemplified best in the balanced accuracy score approaching 100%. We also see that in a much improved recall score; of all of the actual high-risk loans, the model predicted the outcome correctly 99% of the time (up from 91% in Machine Model 1).

For the purposes of revenue, it's better to predict a loan as high-risk and it's actually healthy, than it is for the loan to be predicted as healthy and is actually high-risk. As a result, we should also not be deterred by the lower precision score of high-risk loans in Machine-Learning Model 2 (though the difference is small). Both models would work well here as they both have extremely high precision and recall scores for healthy loans, however, **I would recommend Machine Learning Model 2**, due to its higher recall and accuarcy scores.
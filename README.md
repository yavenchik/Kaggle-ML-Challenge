# Kaggle-ML-Challenge
The purpose of the championship was to train a ML model to predict the bank clients' churn rate in the best way basing on the training dataset. Also the test .csv-file as well as the sample submission one were available. The ROC AUC metric was leveraged as a measure of success.
My approach was to initially use the AutoGluon framework to determine the most suitable model. After installing this tool and handling all the library dependencies, the CatBoost model was chosen as the ultimate tool for the project. 
Later on, I wrote a Python script fine-tuning Catboost for the most accurate prediction configuration. The weakness, though, is insufficient EDA. Nevertheless, the model ran relatively successfully and I managed to reach the Top 7 final scores in this competition. 
The code is structured in the following way:
1. Library import;
2. Loading and processing the data;
3. Setting hyperparameters;
4. Training the final model;
5. Assessing the model with several metrics;
6. Additional cross-validation through model wrapping;
7. Writing the submission file.

As for the chosen hyperparameters, I was playing with them, but the final approach lets CatBoost select the most appropriate ones, so they were set as a range (an array). In order to accelerate the script execution, I included running on GPU. In training the final model, I set the iteration number as 200 and included early stopping at 30 iterations for avoiding overtraining. In the Stratified K-Fold CV, I chose 10 as the split number.
Please see the code for further details.

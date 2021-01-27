# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to assess if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
#### Attribute Information:

##### Input variables:
###### Bank client data:
1 - age (numeric) <br>
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br>
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br>
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br>
5 - default: has credit in default? (categorical: 'no','yes','unknown')<br>
6 - housing: has a housing loan? (categorical: 'no','yes','unknown')<br>
7 - loan: has personal loan? (categorical: 'no','yes','unknown')<br>
###### Related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')<br>
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')<br>
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')<br>
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
###### Other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br>
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br>
14 - previous: number of contacts performed before this campaign and for this client (numeric)<br>
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br>
###### Social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)<br>
17 - cons.price.idx: consumer price index - monthly indicator (numeric)<br>
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)<br>
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)<br>
20 - nr.employed: number of employees - quarterly indicator (numeric)<br>

##### Output variable (desired target):
21 - y - has the client subscribed to a term deposit? (binary: 'yes','no')s
The classification was performed using a Scikit-learn pipeline and an AutoML pipeline. 
## Scikit-learn Pipeline

The scikit-learn pipeline has the following components: 

**Data preparation.** This involves steps like importing, validating, cleaning, wrangling (or "munging"), transforming, normalizing, and staging your data. This step tends to be a large proportion of the work in most ML projects. 

**Training configuration.** A typical training configuration includes steps like parameterization, file paths, logging, and reporting. 

**Training validation.** Training validation involves repeatedly running through your experiment, picking different hardware, compute resources, doing distributed computing, and also monitoring your progress. 

The required dataset is loaded as TabularDataset using the TabularDatasetFactory in the AzureML core which is then converted into a dataframe using pandas library. The loaded dataset undergoes One-Hot Encoding on several attributes to transform and stage the data for training. The dataset is split into test and train with the test set being 20% of the data and at the default random state of 42.

The scikit-learn pipeline uses Logistic Regression which requires the following hyperparameters:<br>
1. **--C** - Inverse of regularization strength.
2. **--max_iter** - Maximum number of iterations to converge

Azure Hyperdrive is used in this pipeline to perform hyperparameter tuning with RandomParameterSampling being the parameter sampler. The required hyperparameters were chosen at random out of the given set of values. 

"--C": choice(0.5, 0.75, 1.0, 1.25), <br>
"--max_iter": choice(10, 50, 100, 200)

A compute cluster with vm_size ```STANDARD_D2_V2``` and 4 maximum nodes is used to run the experiment. The HyperDriveConfig is created with the mentioned samples, estimater and policy along with maximum total runs set to 20 and maximum concurrent runs to 5.

There are three types of sampling in the hyperparameter space:<br>
1. **Random Sampling**<br>
Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space.

2. **Grid Sampling**<br>
Grid sampling supports discrete hyperparameters. Use grid sampling if you can budget to exhaustively search over the search space. Supports early termination of low-performance runs. Performs a simple grid search over all possible values. Grid sampling can only be used with choice hyperparameters. 

3. **Bayesian Sampling**<br>
Bayesian sampling is based on the Bayesian optimization algorithm. It picks samples based on how previous samples performed, so that new samples improve the primary metric. Bayesian sampling is recommended if you have enough budget to explore the hyperparameter space. For best results, we recommend a maximum number of runs greater than or equal to 20 times the number of hyperparameters being tuned.

Random Sampling is chosen as the sampling parameter as it supports both discrete and continuous hyperparameters providing a wider range of possible parameter combinations for the users. Grid sampling supports only discrete hyperparameters and performs an exhaustive search over the parameter space which requires high computational resources. Bayesian Sampling is justified when the maximum runs is greater than equal to 20, also demanding budget enough to withstand. Apart from that Bayesian sampling does not support early termination which is a requirement for our project. So, random sampling is an efficient choice for our dataset.

Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. Bandit Policy with a smaller allowable slack is used for aggressive savings, which means that the running jobs can be terminated by Azure in case of higher priority requirements of resources. Since our project does not need to run continuously, such an aggressive savings policy is sufficient than a conservative savings policy.

## AutoML
The algorithm pipeline with highest accuracy is VotingEnsemble.

A voting ensemble (or a “majority voting ensemble“) is an ensemble machine learning model that combines the predictions from multiple other models. It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble. A voting ensemble works by combining the predictions from multiple models. It can be used for classification or regression. In the case of regression, this involves calculating the average of the predictions from the models. In the case of classification, the predictions for each label are summed and the label with the majority vote is predicted.<br>
Voting ensembles are most effective when:

  - Combining multiple fits of a model trained using stochastic learning algorithms.<br>
  - Combining multiple fits of a model with different hyperparameters.<br>
## Pipeline comparison

The scikit-learn pipeline generated the following parameters with its best run:<br>
```Accuracy:  0.9072837632776934```<br>
```Regularization Rate:  0.75```<br>
```Number of iterations:  100```<br>
Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error. Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions. Consequently, most logistic regression models use one of the following two strategies to dampen model complexity:

  - L2 regularization.<br>
  - Early stopping, that is, limiting the number of training steps or the learning rate.

The scikit-learn Logistic Regression algorithm demands the following parameters:<br>
1. **C** float, default=1.0<br>
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

2. **max_iter** int, default=100 <br>
Maximum number of iterations taken for the solvers to converge.

More information about it can be found at,<br>
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

The VotingEnsemble pipeline in AutoML generated the highest accuracy:<br>
```Accuracy: 0.9166918938065288```

With algorithm weights as:<br>
```weights=[0.23076923076923078,```<br>
        ```0.07692307692307693,```<br>
        ```0.07692307692307693,```<br>
        ```0.15384615384615385,```<br>
        ```0.07692307692307693,```<br>
        ```0.07692307692307693,```<br>
        ```0.07692307692307693,```<br>
        ```0.07692307692307693,```<br>
        ```0.07692307692307693,```<br>
        ```0.07692307692307693]```<br>

The difference in performance between both the pipelines is due to the advanced automation and processing features associated with AutoML. If there is no constraint on the model to be used in the project, then AutoML is the optimized choice to run the experiment as it applies multiple pipelines to obtain an optimized performance. The scikit-learn pipeline was limited to Logistic Regression and tested runs with various hyperparameters, while the AutoML pipeline ran the experiment of different model pipelines with hyperparameter tuning. The metrics and model explanation generated by the AutoML run can be used to improve the scikit-learn pipeline more insightfully. 
## Future work
A limitation of the voting ensemble is that it treats all models the same, meaning all models contribute equally to the prediction. This is a problem if some models are good in some situations and poor in others. In order to improve the model performance in the future:
1. Prevent overfitting

An over-fitted model will assume that the feature value combinations seen during training will always result in the exact same output for the target.<br>
The best way to prevent overfitting is to follow ML best-practices including:

  - Using more training data, and eliminating statistical bias
  - Preventing target leakage
  - Using fewer features
  - Regularization and hyperparameter optimization
  - Model complexity limitations
  - Cross-validation
  
In the context of automated ML, the first three items above are best-practices you implement. The last three bolded items are best-practices automated ML implements by default to protect against over-fitting. In settings other than automated ML, all six best-practices are worth following to avoid over-fitting models.<br>
More information can be found at,<br>
https://docs.microsoft.com/en-us/azure/machine-learning/concept-manage-ml-pitfalls

2. Using wider ranging hyperparameter sampling in the scikit-learn pipeline

3. Performing feature selection by exploring the dataset

By choosing the right features, you can potentially improve the accuracy and efficiency of classification. You typically use only the columns with the best scores to build your predictive model. Columns with poor feature selection scores can be left in the dataset and ignored when you build a model.<br>
For more information,
https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/feature-selection-modules

4. Use the AutoML run "Explanation" tab to view the aggregate feature importance and individual feature importance to perform more insightful feature selection.

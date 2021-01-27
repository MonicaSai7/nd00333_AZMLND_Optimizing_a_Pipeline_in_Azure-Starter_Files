# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
#### Attribute Information:

##### Input variables:
###### bank client data:
1 - age (numeric) <br>
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br>
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br>
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br>
5 - default: has credit in default? (categorical: 'no','yes','unknown')<br>
6 - housing: has housing loan? (categorical: 'no','yes','unknown')<br>
7 - loan: has personal loan? (categorical: 'no','yes','unknown')<br>
###### related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')<br>
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')<br>
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')<br>
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
###### other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br><br>
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br>
14 - previous: number of contacts performed before this campaign and for this client (numeric)<br>
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br>
###### social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)<br>
17 - cons.price.idx: consumer price index - monthly indicator (numeric)<br>
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)<br>
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)<br>
20 - nr.employed: number of employees - quarterly indicator (numeric)<br>

##### Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')s
The classification was performed using a Scikit-learn pipeline and an AutoML pipeline. 
## Scikit-learn Pipeline

The scikit-learn pipeline has the following components: 

**Data preparation.** This involves steps like importing, validating, cleaning, wrangling (or "munging"), transforming, normalizing, and staging your data. This step tends to be a large proportion of the work in most ML projects. 

**Training configuration.** A typical training configuration includes steps like parameterization, file paths, logging, and reporting. 

**Training validation.** Training validation involves repeatedly running through your experiment, picking different hardware, compute resources, doing distributed computing, and also monitoring your progress. 

The required dataset is loaded as TabularDataset using the TabularDatasetFactory in the AzureML core which is then converted into a dataframe using pandas library. The loaded dataset undergoes One-hot encoding on several attributes to transform and stage the data for training. The dataset is split into test and train with test set being 20% of the data and at the default random state of 42.

The scikit-learn pipeline uses Logistic Regression which requires the following hyperparameters:<br>
1. **--C** - Inverse of regularization strength.
2. **--max_iter** - Maximum number of iterations to converge

Azure Hyperdrive is used in this pipeline to perform hyperparameter tuning with RandomParamterSampling being the paramter sampler. The required hyperparamters were chosen at random out of the given set of values. 

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

Random Sampling is chosen as the sampling paramter as it supports both discrete and continuous hyperparamters providing wider range of possible paramter combbination for the users. Grid sampling supports only discrete hyperparamters and performs an exhaustive search over the parameter space which requires high computational resources. Bayesian Sampling is justified when the maximum runs is greater than equal to 20, also demanding budget enough to withstand. Apart from that Bayesian sampling does not support early termination which is a requirement for our project. So, random sampling is an efficient choice for our dataset.

Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. Bandit Policy with a smaller allowable slack is used for aggressive savings, which means that the running jobs can be terminated by Azure in case of higher priority requirements of resources. Since our project does not needs to run continuously, such an agressive savings policy is sufficient than a conservative savings policy.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

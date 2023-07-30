# shallow_vs_dl

## Install

(Using python 10)

Create a venv: `python -m venv env`

Use the environment `source env/bin/activate`

Install the requirements `pip install -r requirements.txt`

## Dataset

### Generate the dataset

To generate the dataset in the default folder run on the git root folder `python dataset/preprocessing.py`

If you want any of the other two data genereation options running `python dataset/preprocessing.py -h` will show the possible options.

### Dataset characteristics

| Total examples | Nº Features | <div>Examples per class <br>(URLLC\eMBB\mMTC)</div> |
|----------------|-------------|----------------------------------------------------|
| | **Training set** | |
| 373391         | 60          | 45\27\28                               |
| | **Test set** | |
| 93348          | 60          | 45\27\28                               |


## Model optimization
The shallow model's optimized hyperparameters are presented in the tables below.
For more information regarding the model's hyperparamerters visit the Scikit-learn website https://scikit-learn.org/ 

#### Logistic Regression
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| C                  | 0.001, 0.01, 0.1, 1, 10       |
| Penalty            | l1, l2, Elasticnet            |
| Solver             | Sag, Saga, Lbfgs              |


#### Support Vector Machine
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| C                  | 0,001, 0.01, 0.1, 1, 10       |
| Kernel             | Linear, RBF                   |

#### K-Nearest Neighbors
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Nº Neighbors       | 3, 5, 7                       |
| Weights            | Uniform, Distance             |

#### Decision Tree
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Criterion          | Gini, Entropy                 |
| Max depth          | 3, 5, 7, 9, 15, 20, 25, 30    |
| Max features       | auto, sqrt, log2              |

#### Gaussian Naive Bayes
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Variable smoothing | logspace(0, -9)               |

#### Random forest
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Nº estimators      | 2, 3, 5, 10, 20, 50, 100, 200 |
| Max depth          | 3, 5, 7, 9, 15, 20, 25, 30    |
| Max features       | auto, sqrt, log2              |

#### AdaBoost
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Nº estimators      | 2, 3, 5, 10, 20, 50, 100, 200 |

#### Gradient boosting
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Nº estimators      | 2, 3, 5, 10, 20, 50, 100, 200 |
| Max depth          | 3, 5, 7, 9, 15, 20, 25, 30    |
| Max features       | auto, sqrt, log2              |

## Run experiments

If the preprocessed dataset is in the default folder:

In the root folder run `python dl_models/train.py` or `python shallow_models/train.py`

If the dataset is on a different folder run `python dl_models/train.py -h` to see the arguments on how to define the dataset folder.

The seed used can also be changed as shown in the possible arguments.

The shallow models are automatically go through the grid search when running the experiments.

## Experiments' results

### Results for the Neural Networks

#### Using patience=3 - Not used
| Model name | Train time | Infer time | ACC | F1  | MCC |
| ---------- | ---------- | ---------- | --- | --- | --- |
| DNN        | 188.47±7.1   | 3.51±0.26  | 1.0 | 1.0 | 1.0 |
| AUTODEEPSLICE | 198.69±7.96  | 3.26±0.47  | 1.0 | 1.0 | 1.0 |
| DEEPSLICE  | 189.67±3.93  | 3.85±0.32  | 1.0 | 1.0 | 1.0 |

#### Using patience=0 - Used
| Model name | Train time | Infer time | ACC | F1  | MCC |
| ---------- | ---------- | ---------- | --- | --- | --- |
| 0          | 158.28 ±4.75  | 3.45±0.49  | 1.0 | 1.0 | 1.0 |
| 1          | 86.24  ±4.27  | 3.54±0.41  | 1.0 | 1.0 | 1.0 |
| 2          | 157.95 ±2.52  | 3.63±0.38  | 1.0 | 1.0 | 1.0 |


### Results for the Shallow models

#### Hyperparameter optmization results

| Model             | Values                                             |
|-------------------|----------------------------------------------------|
| LR                | C: 0.001, Penalty: l2, Solver: Sag                 |
| SVM               | C:10, Kernel: linear                               |
| k-NN              | Nº neighbors:3, Weights: Distance                  |
| DT                | Criterion: gini, Max depth: 30, Max features: sqrt |
| GausianNB         | Variable smoothing: 0.00000018738174228604         |
| RF                | Nº estimators:2, Max depth: 3, Max features: log2  |
| AdaBoost          | Nº estimators: 3                                   |
| Gradient Boosting | Nº estimators:5, Max depth: 7, Max features: log2  |

#### Performance results

| Model name | Train time | Infer time | ACC | F1  | MCC |
| ---------- | ---------- | ---------- | --- | --- | --- |
| LOG        | 5.18±0.1   | 0.02±0.01  | 1.0 | 1.0 | 1.0 |
| SVM        | 2.74±0.0   | 0.19±0.0   | 1.0 | 1.0 | 1.0 |
| NB         | 0.33±0.0   | 0.1 ±0.0   | 1.0 | 1.0 | 1.0 |
| DT         | 0.35±0.02  | 0.01±0.0   | 1.0 | 1.0 | 1.0 |
| RF         | 0.28±0.23  | 0.04±0.01  | 1.0 | 1.0 | 1.0 |
| ABC        | 1.54±0.06  | 0.04±0.0   | 1.0 | 1.0 | 1.0 |
| GBC        | 4.36±0.2   | 0.06±0.0   | 1.0 | 1.0 | 1.0 |
| KNN        | 0.02±0.0   | 845.21±3.13  | 1.0 | 1.0 | 1.0 |

### Acelleration comparison

|  | Training acceleration	|  |	
| ---------- |	---------- | ---------- |
| Models	| Values	| Acceleration |
| Baseline | 157.95 | -- |
|AutoDEEPSLICE |	86.24 | 45.4 |
|DNN |	158.28 | -0.21 |
|LOG |	5.18 | 96.72 |
|SVM |	2.74 | 98.27 |
|NB |	0.33 | 99.79 |
|DT |	0.35 | 99.78 |
|RF |	0.28 | 99.82 |
|ABC |	1.54 | 99.03 |
|GBC |	4.36 | 97.24 |
|KNN |	0.02 | 99.99 |

|  | Inference acceleration |  |		
| ---------- |	---------- | ---------- |
| Models	| Values	| Acceleration |
|Baseline	| 3.85 |	-- |
|AutoDEEPSLICE 	| 3.26 |	15.32 | 
|DNN	| 3.51 |	8.83 | 
|LOG	| 0.02 |	99.48 | 
|SVM	| 0.19 |	95.06 | 
|NB	| 0.1 |	97.4 | 
|DT	| 0.01 |	99.74 | 
|RF	| 0.04 |	98.96 | 
|ABC	| 0.04 |	98.96 | 
|GBC	| 0.06 |	98.44 | 
|KNN	| 845.21 |	-21853.51 | 
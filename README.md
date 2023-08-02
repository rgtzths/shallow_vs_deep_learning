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

| Model name | Train time | Infer time | ACC | F1  | MCC |
| ---------- | ---------- | ---------- | --- | --- | --- |
| DEEPSLICE  | 157.8±4.36  | 3.53±0.23  | 1.0 | 1.0 | 1.0 |
| autoDEEPSLICE | 85.8±1.76  | 3.37±0.14  | 1.0 | 1.0 | 1.0 |
| DNN        | 157.9±3.21  | 3.55±0.43  | 1.0 | 1.0 | 1.0 |


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
| LOG        | 4.74±0.11  | 0.02±0.0   | 1.0 | 1.0 | 1.0 |
| SVM        | 2.76±0.01  | 0.19±0.0   | 1.0 | 1.0 | 1.0 |
| KNN        | 0.01±0.0   | 855.78±1.08  | 1.0 | 1.0 | 1.0 |
| NB         | 0.35±0.0   | 0.1 ±0.0   | 1.0 | 1.0 | 1.0 |
| DT         |  0.4±0.01  | 0.01±0.0   | 1.0 | 1.0 | 1.0 |
| RF         | 0.24±0.01  | 0.04±0.0   | 1.0 | 1.0 | 1.0 |
| ABC        | 1.49±0.03  | 0.04±0.0   | 1.0 | 1.0 | 1.0 |
| GBC        | 4.39±0.17  | 0.06±0.0   | 1.0 | 1.0 | 1.0 |


### Acelleration comparison

|  | Training acceleration	|  |	
| ---------- |	---------- | ---------- |
| Models	|    Values	| Acceleration |
| Baseline |        157.80 | -- |
|AutoDEEPSLICE |	85.80  | 45.63 |
|DNN |	            157.90 | -0.06 |
|LOG |	            4.74  | 97.00 |
|KNN |	            0.01  | 99.99 |
|SVM |	            2.76 | 98.25 |
|NB |	            0.35  | 99.78 |
|DT |	            0.40   | 99.75 |
|RF |	            0.24  | 99.85 |
|ABC |	            1.49  | 99.06 |
|GBC |	            4.39  | 97.22 |


|  | Inference acceleration |  |		
| ---------- |	---------- | ---------- |
| Models	|       Values	| Acceleration |
|Baseline	|       3.53 |	-- |
|AutoDEEPSLICE 	|   3.37 |	    4.53 | 
|DNN	|           3.55 |	    -0.57 | 
|LOG	|           0.02 |	    99.43 | 
|KNN	|           855.78 |	    -24143.06 | 
|SVM	|           0.19 |	 94.62 | 
|NB	|               0.10 |	    97.17 | 
|DT	|               0.01 |	    99.72 | 
|RF	|               0.04 |	    98.87 | 
|ABC	|           0.04 |	    98.87 | 
|GBC	|           0.06 |	    98.30 | 


## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

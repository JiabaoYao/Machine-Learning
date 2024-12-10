# CSE 574 Project3 Report

_Members in Group 6: Jiabao Yao(50602483), Han Li(50993977)_

## Problem 1 Logistic Regression

The Logistic Regression model for 10 classes permances very well with high trainging, validation, and test accuracy more than 90%.  
![](accuracy_problem1.png)

To evaluate the performance of this model, we recorded the accuracy for both the training set and testing set across each category. As shown in the graph below, all classes achieve an accuracy of over 90%. Notably, classes 0 and 1 exhibit better performance, while classes 8 and 9 have slightly higher errors compared to the others. This observation aligns with the fact that lower-numbered classes often represent simpler patterns, making them easier to learn. Additionally, the testing set errors are consistently higher than the training set errors, which may be attributed to overfitting. This occurs when the model fits the training data too closely, resulting in higher accuracy on the training set but reduced generalization to unseen data in the testing set.
![](error_problem1.png)

## Problem 2 Multi-Class Logistic Regression

Compared to one-vs-all strategy, multi-class strategy has higher accuracy over 90%.  
![](accuracy_problem2.png)  
The overall performance of the multi-class logistic regression model is outstanding, achieving over 90% accuracy across all three datasets. The training set exhibits slightly higher accuracy compared to the testing and validation sets, which is expected since the model is directly optimized on the training data. However, the slightly higher accuracy on the training set could also indicate mild overfitting. Nonetheless, the above 90% accuracy on both the validation and testing datasets demonstrates that the model generalizes well to unseen data.

Multi-class logistic regression slightly outperforms the one-vs-all approach across all metrics. This improvement can be attributed to the use of the softmax function, which evaluates all classes simultaneously, resulting in more consistent and balanced predictions. In contrast, the one-vs-all strategy trains separate binary classifiers for each class, which can lead to conflicts in predictions and slightly reduced performance.

## Problem 3 SVM

We test with these different parameters:

1. Using linear kernel (all other parameters are kept default).
2. Using radial basis function with value of gamma setting to 1 (all other parameters are kept default).
3. Using radial basis function with value of gamma setting to default (all other parameters are kept
   default).
   And get such result

When we set `gamma` to 1, we get very low accuracy.

```
[linear]Validation set Accuracy:93.64%
[linear]Testing set Accuracy:93.78%
[rbf-1]Validation set Accuracy:21.43%
[rbf-1]Testing set Accuracy:22.71%
[rbf]Validation set Accuracy:97.89%
[rbf]Testing set Accuracy:97.87%

```

![](svm1.png)

After that we test with different `C`, when `C` is set to 10, we can have the highest accuracy.
![](svm2.png)

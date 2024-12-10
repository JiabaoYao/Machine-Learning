linear_validation_accuracy=93.64
linear_test_accuracy=93.78

rbf_gamma_1_validataionaccuracy=21.43
rbf_gamma_1_test_accuracy=22.71

rbf_gamma_default_validataion_accuracy=97.89
rbf_gamma_default_test_accuracy=97.87

C_list=[1,10,20,30,40,50,60,70,80,90,100]
C_validation=[0.9787, 0.9834, 0.9831, 0.9831, 0.9831, 0.9831, 0.9831, 0.9831, 0.9831, 0.9831, 0.9831]
C_validation = [i * 100 for i in C_validation]
C_result=[0.9789, 0.9845, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844]
C_result = [i * 100 for i in C_result]


import matplotlib.pyplot as plt
import numpy as np


x_list = ['linear', 'rbf-with gamma=1', 'rbf-with default params']
validation_list = [linear_validation_accuracy, rbf_gamma_1_validataionaccuracy, rbf_gamma_default_validataion_accuracy]
test_list = [linear_test_accuracy, rbf_gamma_1_test_accuracy, rbf_gamma_default_test_accuracy]
plt.ylabel('Accuracy')
plt.xlabel('Kernel Params')

x = np.arange(len(x_list))
x_group1 = x
x_group2 = x + 0.4
plt.bar(x_group1, validation_list, width=0.4, align='center', label='Validation')
plt.bar(x_group2, test_list, width=0.4, align='center',label='Testing')
plt.xticks(x + 0.2, x_list)
plt.legend(fontsize=12) 

plt.savefig('svm1.png')

plt.clf()

x_list = [ 'C=1', 'C=10', 'C=20', 'C=30', 'C=40', 'C=50', 'C=60', 'C=70', 'C=80', 'C=90', 'C=100']
validation_list=  C_validation
test_list =  C_result
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.plot(x_list, validation_list, label='Validation')
plt.plot(x_list, test_list, label='Testing')
plt.legend(fontsize=12) 

plt.savefig('svm2.png')


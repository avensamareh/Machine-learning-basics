#!/usr/bin/env python
# coding: utf-8

# ### Loading library
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


# ### Read data
data = pd.read_csv('model_outcome.csv')
#data.head()
#data.shape

true_class = data['class']  # true class labels
predicted_values1 = data['predicted_prob'] # predicted class labels

# Converting predicted values into classes using threshold
threshold = 0.5
predicted_class1 = np.zeros(predicted_values1.shape)
predicted_class1[predicted_values1>threshold]=1
pred_pos = predicted_class1[predicted_class1 == 1]
pred_neg = predicted_class1[predicted_class1 == 0]
#predicted_class1.shape

thresholds = np.linspace(1,0.5,1000)

def perf_measure(true_class, predicted_values1):
    ROC = np.zeros((1000,2))
    for i in range(1000):
        t = thresholds[i]
        #t = threshold
        TP_t = np.logical_and( predicted_values1 > t, true_class==1 ).sum()
        TN_t = np.logical_and( predicted_values1 <=t, true_class==0 ).sum()
        FP_t = np.logical_and( predicted_values1 > t, true_class==0 ).sum()
        FN_t = np.logical_and( predicted_values1 <=t, true_class==1 ).sum()
        
        # Specificity or true negative rate
        TNR = TN_t/(TN_t+FP_t)
        
        # Precision or positive predictive value
        PPV = TP_t/(TP_t+FP_t)
        
        # Negative predictive value
        NPV = TN_t/(TN_t+FN_t)
        
        # False negative rate
        FNR = FN_t/(TP_t+FN_t)
        
        # False discovery rate
        FDR = FP_t/(TP_t+FP_t)
        
        # Overall accuracy
        ACC = (TP_t+TN_t)/(TP_t+FP_t+FN_t+TN_t)
        
        # Compute false positive rate for current threshold
        FPR_t = FP_t / float(FP_t + TN_t)
        
        # Sensitivity, hit rate, recall, or true positive rate
        TPR_t = TP_t / float(TP_t + FN_t)
        
        ROC[i,1] = TPR_t
        ROC[i,0] = FPR_t
    
    return TP_t, FN_t, FP_t , TN_t, PPV, TNR, TPR_t, FPR_t,  ACC, ROC


# ### 1. Manually calculate the sensitivity and specificity of the model, using a predicted_prob threshold of greater than or equal to .5.
TP_t, FN_t, FP_t , TN_t, PPV, TNR, TPR_t, FPR_t,  ACC, ROC = perf_measure(true_class, predicted_values1 )
print('sensitivity: ',TPR_t)
print('specificity: ',TNR)
print('accuracy: ', ACC)


# ### 2. Manually calculate the Area Under the Receiver Operating Characteristic Curve.
T = TP_t/(2*(TP_t + FN_t))
U = TN_t/(2*(FP_t + TN_t))
AUC = T + U
print('AUC: ',AUC)

# ### 3. Visualize the Receiver Operating Characterstic Curve.
plt.plot(ROC[:,0],ROC[:,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')


# ## Validate manual results via sklearn library
tn, fp, fn, tp = confusion_matrix(true_class,predicted_class1).ravel()
specificity = tn / (tn+fp)
precision = tp/(tp+fp)
sensitivity = tp / float(tp + fn)
print('sensitivity: ',sensitivity)
print('precision',precision)
print('specificity', specificity)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve( true_class,predicted_class1)
    roc_auc[i] = auc(fpr[i], tpr[i])

print roc_auc_score( true_class,predicted_class1)

plt.show()
 




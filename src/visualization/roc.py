import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from joblib import load


classifier_knn = load("/ml/new/knn_model_selected_sg_genes.joblib")
classifier_svm = load("/ml/new/best_svm_model.joblib")
classifier_rf = load("/ml/new/selected_features_model.joblib")

test_features = pd.read_csv("/ml/new/test_features_80_20.csv")
test_labels = pd.read_csv("/ml/new/test_labels_80_20.csv")

relevant_features = [
    'ATOH1', 'GRM8', 'PCDH8', 'CEL', 'CFTR', 'ENGASE', 'GP2', 'C5orf52', 'SATB2', 'CDX2',
    'CYP2B6', 'NKD1', 'SMAD2', 'MTOR', 'NFE2L2', 'RB1', 'KEAP1', 'TERT', 'RASA1', 'CDC73',
    'CTNNA1', 'ERBB4', 'CD8A', 'PRF1', 'GZMA', 'GZMB', 'CX3CL1', 'CXCL9', 'CXCL10', 'IFNG',
    'IL1B', 'LAG3', 'CTLA4', 'CD274', 'PDCD1', 'TIGIT', 'IDO1', 'PDCD1LG2', 'VEGFA', 'VEGFB',
    'VEGFC', 'VEGFD'
]

test_features_relevant = test_features[relevant_features]


positive_class = 'MSI-H'
test_labels_binary = label_binarize(test_labels['msi_status'], classes=[positive_class])

probs_positive_test_knn = classifier_knn.predict_proba(test_features_relevant)[:, 0]
probs_positive_test_svm = classifier_svm.decision_function(test_features_relevant)[:, 0]
probs_positive_test_rf = classifier_rf.predict_proba(test_features_relevant)[:, 0]


fpr_knn_test, tpr_knn_test, _ = roc_curve(test_labels_binary, probs_positive_test_knn)
roc_auc_knn_test = auc(fpr_knn_test, tpr_knn_test)
fpr_svm, tpr_svm, _ = roc_curve(test_labels_binary, probs_positive_test_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)
fpr_rf, tpr_rf, _ = roc_curve(test_labels_binary, probs_positive_test_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)


plt.figure(figsize=(10, 6))
plt.plot(fpr_knn_test, tpr_knn_test, color='green', lw=2, label='KNN (AUC = %0.2f)' % roc_auc_knn_test)
plt.plot(fpr_svm, tpr_svm, color='orange', lw=2, label='SVM (AUC = %0.2f)' % roc_auc_svm)
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='RF (AUC = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC curve', fontsize=18)
plt.legend(loc='lower right', fontsize=18)
plt.show()

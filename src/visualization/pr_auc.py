import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from joblib import load
from sklearn.preprocessing import label_binarize

classifier_knn = load("/ml/new/knn_model_selected_sg_genes.joblib")
classifier_rf = load("/ml/new/selected_features_model.joblib")
classifier_svm = load("/ml/new/best_svm_model.joblib")

relevant_features = [
    'ATOH1', 'GRM8', 'PCDH8', 'CEL', 'CFTR', 'ENGASE', 'GP2', 'C5orf52', 'SATB2', 'CDX2',
    'CYP2B6', 'NKD1', 'SMAD2', 'MTOR', 'NFE2L2', 'RB1', 'KEAP1', 'TERT', 'RASA1', 'CDC73',
    'CTNNA1', 'ERBB4', 'CD8A', 'PRF1', 'GZMA', 'GZMB', 'CX3CL1', 'CXCL9', 'CXCL10', 'IFNG',
    'IL1B', 'LAG3', 'CTLA4', 'CD274', 'PDCD1', 'TIGIT', 'IDO1', 'PDCD1LG2', 'VEGFA', 'VEGFB',
    'VEGFC', 'VEGFD'
]

test_features = pd.read_csv("/ml/new/test_features_80_20.csv")
test_labels = pd.read_csv("/ml/new/test_labels_80_20.csv")

positive_class = 'MSI-H'
test_labels_binary = label_binarize(test_labels['msi_status'], classes=[positive_class])

probs_positive_test_knn = classifier_knn.predict_proba(test_features[relevant_features])[:, 0]
probs_positive_test_rf = classifier_rf.predict_proba(test_features[relevant_features])[:, 0]
probs_positive_test_svm = classifier_svm.decision_function(test_features[relevant_features])

precision_knn, recall_knn, _ = precision_recall_curve(test_labels_binary, probs_positive_test_knn)
pr_auc_knn = auc(recall_knn, precision_knn)

precision_rf, recall_rf, _ = precision_recall_curve(test_labels_binary, probs_positive_test_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

positive_scores_svm = probs_positive_test_svm[:, 0]
precision_svm, recall_svm, _ = precision_recall_curve(test_labels_binary[:, 0], positive_scores_svm)
pr_auc_svm = auc(recall_svm, precision_svm)

plt.figure(figsize=(10, 6))
plt.plot(recall_knn, precision_knn, color='green', lw=2, label='KNN (PR-AUC = %0.2f)' % pr_auc_knn)
plt.plot(recall_rf, precision_rf, color='blue', lw=2, label='RF (PR-AUC = %0.2f)' % pr_auc_rf)
plt.plot(recall_svm, precision_svm, color='red', lw=2, label='SVM (PR-AUC = %0.2f)' % pr_auc_svm)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision' , fontsize=18)
plt.title('Precision-Recall Curve', fontsize=18)
plt.legend(loc='lower left', fontsize=18)
plt.grid(True)
plt.show()

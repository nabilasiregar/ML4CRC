import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from joblib import load


classifier_rf = load("/ml/new/all_features_model.joblib")
classifier_knn = load("/ml/new/knn_model_smote.joblib")


test_features = pd.read_csv("/ml/new/test_features_80_20.csv")
test_labels = pd.read_csv("/ml/new/test_labels_80_20.csv")

positive_class = 'MSI-H'
test_labels_binary = label_binarize(test_labels['msi_status'], classes=[positive_class])

probs_positive_test_rf = classifier_rf.predict_proba(test_features)[:, 0]
probs_positive_test_knn = classifier_knn.predict_proba(test_features)[:, 0]

fpr_rf, tpr_rf, _ = roc_curve(test_labels_binary, probs_positive_test_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
fpr_knn, tpr_knn, _ = roc_curve(test_labels_binary, probs_positive_test_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)


plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='RF (AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_knn, tpr_knn, color='green', lw=2, label='KNN (AUC = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC curve', fontsize=18)
plt.legend(loc='lower right', fontsize=18)
plt.show()

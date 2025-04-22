{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62b10d1a-2254-49fb-a3df-7eabbe3905fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n",
    "\n",
    "def get_clf_eval(y_true, y_pred, y_proba):\n",
    "    \"\"\"\n",
    "    Compute classification metrics including accuracy, precision, recall, specificity, F1 score, AUC, and balanced accuracy.\n",
    "    \"\"\"\n",
    "    confusion = confusion_matrix(y_true, y_pred)\n",
    "    accuracy = accuracy_score(y_true, y_pred)\n",
    "    precision = precision_score(y_true, y_pred)\n",
    "    recall = recall_score(y_true, y_pred)\n",
    "    f1 = f1_score(y_true, y_pred)\n",
    "    roc_auc = roc_auc_score(y_true, y_proba)\n",
    "    specificity = confusion[0,0] / (confusion[0,0] + confusion[0,1])\n",
    "    balanced_acc = (recall + specificity) / 2\n",
    "\n",
    "    print('Confusion Matrix:')\n",
    "    print(confusion)\n",
    "    print('Accuracy: {:.4f}, Precision: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}, F1 Score: {:.4f}, AUC: {:.4f}, Balanced Accuracy: {:.4f}'\n",
    "          .format(accuracy, precision, recall, specificity, f1, roc_auc, balanced_acc))\n",
    "\n",
    "    return accuracy, precision, recall, specificity, f1, roc_auc, balanced_acc\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "onekernel",
   "language": "python",
   "name": "onefirst"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

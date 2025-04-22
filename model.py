{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0dc97408-db8c-4edd-ab16-403e099862c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from xgboost import XGBClassifier\n",
    "from lightgbm import LGBMClassifier\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "\n",
    "def get_models():\n",
    "    \"\"\"\n",
    "    Initialize three classifiers: XGBoost, LightGBM, and Gradient Boosting.\n",
    "    \"\"\"\n",
    "    model1 = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, scale_pos_weight=5)\n",
    "    model2 = LGBMClassifier(n_estimators=300, learning_rate=0.01, max_depth=7, min_split_gain=0.01, scale_pos_weight=5)\n",
    "    model3 = GradientBoostingClassifier(n_estimators=600, learning_rate=0.01, max_depth=4, min_samples_split=2)\n",
    "    return model1, model2, model3"
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

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ce0ac27-fa50-45f5-8201-dbbf3197c849",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import os\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Set dataset path (Modify this if necessary)\n",
    "DATA_PATH = os.getenv('DATASET_PATH', 'your/private/dataset/path')\n",
    "\n",
    "def load_data():\n",
    "    \"\"\"\n",
    "    Load dataset from a private source.\n",
    "    Modify the file path as needed.\n",
    "    \"\"\"\n",
    "    df = pd.read_table(os.path.join(DATA_PATH, 'data.txt'), sep=',', low_memory=False)\n",
    "\n",
    "    # Drop unnecessary columns\n",
    "    X_features = df.drop(['institution_id', 'patient_id', 'visit_date', 'visit_time',\n",
    "                          'survive', 'in_hospital_deceased', 'in_hospital_survive', 'ISS'], axis=1).values\n",
    "    y_label = df['survive'].values\n",
    "    return X_features, y_label\n",
    "\n",
    "def preprocess_data(X, y):\n",
    "    \"\"\"\n",
    "    Split data into training and testing sets.\n",
    "    \"\"\"\n",
    "    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=77)\n"
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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

TRAIN_PATH = "data/train/train.csv"
TEST_PATH = "data/test/test.csv"

# We load the main csv for training and testing
data_train = pd.read_csv(filepath_or_buffer=TRAIN_PATH)
data_test = pd.read_csv(filepath_or_buffer=TEST_PATH)

# Check the class distribution for the target variable in the train set

target_distribution_train = data_train["AdoptionSpeed"].value_counts()

for key, amount in zip(target_distribution_train.index, target_distribution_train.values):
    print(f"Class {key}: {amount} --> {round((amount/data_train.shape[0])*100, 2)}%")


print(data_train.dtypes)

print(f"Number of training samples: {data_train.shape}")

sns.countplot(x='AdoptionSpeed', data=data_train)

plt.title('AdoptionSpeed distribution')
plt.xlabel('Adoption Speed')
plt.ylabel('Number of Pets')
plt.savefig('AdoptionSpeed_distribution.png') 
import pandas as pd

TRAIN_PATH = "data/train/train.csv"
TEST_PATH = "data/test/test.csv"

#* We load the main csv for training and testing
data_train = pd.read_csv(filepath_or_buffer=TRAIN_PATH)
data_test = pd.read_csv(filepath_or_buffer=TEST_PATH)

#* Check the class distribution for the target variable in the train set
target_distribution_train = data_train["AdoptionSpeed"].value_counts()

for key, amount in zip(target_distribution_train.index, target_distribution_train.values):
    print(f"Class {key}: {amount} --> {round((amount/data_train.shape[0])*100, 2)}%")
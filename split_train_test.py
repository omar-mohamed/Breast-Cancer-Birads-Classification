import pandas as pd
import os
import numpy as np

dataset_df = pd.read_csv('./data/all_data.csv')

number_of_testing_cases=20
shuffle=True

if shuffle:
    dataset_df = dataset_df.sample(frac=1., random_state=np.random.randint(1,100))

training_df=dataset_df.head(-number_of_testing_cases)

print(training_df.head())

testing_df=dataset_df.tail(number_of_testing_cases)

print(testing_df.head())

training_df.to_csv(os.path.join("./data","training_set.csv"), index=False)

testing_df.to_csv(os.path.join("./data","testing_set.csv"), index=False)
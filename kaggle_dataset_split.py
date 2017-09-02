# __author__ Mhttx.
#

"""split the train dataset training.csv into training set and validation set
"""

import pandas as pd 
import numpy as np 
from sklearn.utils import shuffle

def load_drop(filename):
	df = pd.read_csv(filename, header=0)
	print('Initial train size:', df.values.shape[0])

	df = df.dropna()
	print('After drop NaNs:', df.values.shape[0])
	return df

def split(original_df, validation_set_path, training_set_path, validation_propotion=0.2):

	original_df = shuffle(original_df) # shuffle the original set

	validation_size = int(original_df.values.shape[0] * validation_propotion)
	validation_df = original_df[:validation_size]
	training_df = original_df[validation_size:]

	assert validation_df.values.shape[0] + training_df.values.shape[0] == original_df.values.shape[0]
	print('split train size:', training_df.values.shape[0])
	print('split validatiopn size:', validation_df.values.shape[0])
	validation_df.to_csv(validation_set_path, index=False)
	training_df.to_csv(training_set_path, index=False)
	

if __name__ == '__main__':

	df = load_drop('/media/mhttx/F/project_developing/kaggle_facial_keypoint_dataset/training.csv')
	split(df, 
		validation_set_path='/media/mhttx/F/project_developing/kaggle_facial_keypoint_dataset/training_set.csv', 
		training_set_path='/media/mhttx/F/project_developing/kaggle_facial_keypoint_dataset/validation_set.csv',
		validation_propotion=0.3)

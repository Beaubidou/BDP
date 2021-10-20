import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from dataset import Dataset

def train_test_sample_split(dataset, start_date="2020-05-24", end_date="2021-05-24", test_ratio=1/2, multi_z=False):
	"""
    Separate the provided dataset into a train set and a test set
    :param start_date: [str ("YYYY-MM-DD")] 
    :param end_date: [str ("YYYY-MM-DD")] 
    :param test_ratio: [float < 1] the test/data set ratio 
	:param multi_z: [bool] True if multi-zone, False if single-zone
    :return: X_train, X_test, y_train, y_test
    """
	ds = dataset.set_index('time')
	ds = ds[start_date : end_date]
	ds = ds.reset_index()

	if multi_z:
		y = pd.concat([ds['time'], ds['current_value_bathroom'], ds['current_value_kitchen'], 
						ds['current_value_bedroom1'], ds['current_value_bedroom2'],
						ds['current_value_bedroom3'], ds['current_value_diningroom'],
						ds['current_value_livingroom']], axis=1)
		X = ds

	else:
		y = pd.concat([ds['time'], ds['current_value_house']], axis=1)
		X = ds

	return train_test_split(X, y, test_size=test_ratio)



if __name__ == "__main__":

	# Test

	df = Dataset().data

	start_date = "2020-06-24"
	end_date = "2020-12-27"
	test_ratio = 2/3
	
	X_train, X_test, _, _ = train_test_sample_split(df, start_date, end_date, test_ratio, multi_z=False)

	print("train set : ")
	print(X_train)

	print("test set : ")
	print(X_test)

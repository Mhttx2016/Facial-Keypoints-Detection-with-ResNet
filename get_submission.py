# __author__ Mhttx


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.externals import joblib


def generate_submission():
    '''genlserate submission csv file for kaggle
    '''
    num_keypoint = 15
    test_labels = np.load('test_predict.npy') # with shape[batch_size, num_keypoint, 2]
    test_labels = test_labels.reshape((test_labels.shape[0], 2 * num_keypoint))
    print('test_labels:', test_labels.shape)

    lookup_table = pd.read_csv('/media/mhttx/F/project_developing/kaggle_facial_keypoint_dataset/IdLookupTable.csv')

    values = []
    cols = joblib.load('cols.pkl')

    for index, row in lookup_table.iterrows():
        print('index:', index, 'row:', row)
        values.append((
                row['RowId'],
                test_labels[row.ImageId - 1][np.where(cols == row.FeatureName)[0][0]],
                ))

    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    generate_submission()


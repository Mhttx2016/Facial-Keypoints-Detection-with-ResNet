# Copyright 2016 The TensorFlow Authors. All Rights Reserved. Modified by Mhttx
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""kaggle facial keypoint dataset input module.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib

def build_input(data_path, batch_size=128, mode='train'):
    """kaggle facial keypoint dataset image and labels.
    NOTE: the dataset annotations are inconsistent(https://www.reddit.com/r/MachineLearning/comments/2pknvo/tutorial_using_convolutional_neural_nets_to/)

    Args:
        dataset: Either 'train' or 'eval'.
        data_files: a list of Filenames for data.
        batch_size: Input batch size.
        mode:'train' or 'eval'
    Returns:
        images: Batches of images. [batch_size, image_size, image_size, 3]
        labels: Batches of labels. [batch_size, num_classes]
        Raises:
    ValueError: when the specified dataset is not supported.
    """

    # image size must be fixed, if not, resize it before read
    image_size = 96
    depth = 1
    num_keypoint = 15

    data_files = tf.gfile.Glob(data_path)
    print('data_files:', data_files)
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # read label as float32 and image data as a string, NaN labels are set to -96.0, normalized to -1.0
    record_defaults = [[float(-image_size)] for _ in range(num_keypoint*2)] + [[""]] 
    # print('record_defaults:', record_defaults)
    cols = tf.decode_csv(value, record_defaults=record_defaults)
    label = tf.stack(cols[:-1])
    image = [cols[-1]]
    image = tf.string_split(image, ' ')
    image = tf.string_to_number(tf.sparse_tensor_to_dense(image, '0'),tf.int32)
    image = tf.reshape(image, [image_size, image_size,depth])

    # preprocess image
    # if mode == 'train':
        # image = tf.image.resize_image_with_crop_or_pad( # # the coordinates should also be adjusted
        # image, image_size+4, image_size+4)
        # image = tf.random_crop(image, [image_size, image_size, depth])
        # image = tf.image.random_flip_left_right(image) # the coordinates should also be flipped
        # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
        # image = tf.image.random_brightness(image, max_delta=63. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        # image = tf.image.per_image_standardization(image) #(x - mean) / adjusted_stddev
    # else:
        # image = tf.image.resize_image_with_crop_or_pad(
        #     image, image_size, image_size)
        # image = tf.image.per_image_standardization(image)

    image = tf.image.per_image_standardization(image)

    # preprocess label
    # label = (label - (image_size // 2)) / (image_size / 2)
    label = label / float(image_size)

    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], 
        batch_size=batch_size,
        capacity=16 * batch_size,
        min_after_dequeue=8*batch_size,
        num_threads=2)
    
    assert len(image_batch.get_shape()) == 4
    assert image_batch.get_shape()[0] == batch_size
    assert image_batch.get_shape()[-1] == depth
    assert len(label_batch.get_shape()) == 2
    assert label_batch.get_shape()[0] == batch_size
    assert label_batch.get_shape()[1] == num_keypoint * 2

    # Display the training images in the visualizer.
    tf.summary.image('images', image_batch)

    # ***********Debug Start************

    # with tf.Session() as sess:
    #     # start populating the filename queue
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord = coord)

    #     for _ in range(1000):
    #         examples, labels = sess.run([image_batch, label_batch])
    #         print('####'*20)
    #         print(labels)
    #         # print('****'*20)
    #         # print(examples)

    # coord.request_stop()
    # coord.join(threads)

    # ***********Debug End************

    return image_batch, label_batch


def test_input(filename, labels_exist=False):
    '''read .csv for test 
        return:
            images: with shape[batch_size, image_size, image_size, depth]
            labels: with shape[batch_size, num_keypoint, 2]
    '''
    image_size = 96
    num_keypoint = 15
    depth = 1 

    df = pd.read_csv(filename)
    cols = df.columns[:-1]
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df = df.dropna()
    images = np.vstack(df['Image'])
    images = images.reshape(-1, image_size, image_size, depth)
    print('size:', images.shape[0])

    labels = None
    if labels_exist:
        labels = df[cols].values.reshape(-1, num_keypoint, 2)
        joblib.dump(cols, 'cols.pkl', compress=3) # save colum features for generate submission
    # print('pppp:', labels[0])
    return images, labels






if __name__ == '__main__':

    # ***********Debug Start************
    build_input(
        batch_size=4,
        data_path='/media/mhttx/F/project_developing/kaggle_facial_keypoint_dataset/training.csv')
    # ***********Debug End************
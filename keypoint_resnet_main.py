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


"""ResNet Train/Eval module for keypoint detection
"""

import time
import sys
import numpy as np
import keypoint_resnet_model 
import tensorflow as tf
from PIL import Image 
import matplotlib.pyplot as plt
import cv2

from kaggle_facial_keypoint_input import *

counter = 1
FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'demo', 'train or eval or test or demo.')

tf.app.flags.DEFINE_string('train_data_path', '/media/mhttx/F/project_developing/kaggle_facial_keypoint_dataset/training_set*',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '/media/mhttx/F/project_developing/kaggle_facial_keypoint_dataset/validation_set.csv',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_string('test_data_path', '/media/mhttx/F/project_developing/kaggle_facial_keypoint_dataset/test.csv',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 96, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '/media/mhttx/F/project_developing/models-master/resnet/keypoint_resnet_model/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '/media/mhttx/F/project_developing/models-master/resnet/keypoint_resnet_model_80000/eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 80,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', True,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '/media/mhttx/F/project_developing/models-master/resnet/keypoint_resnet_model_80000',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')

tf.app.flags.DEFINE_integer('num_keypoint', 15,
                            'Number of keypoints to be detected')

tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch_size')

tf.app.flags.DEFINE_string('demo_image_path', 'timg4.jpg',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')


def train(hps):
    """Training loop."""
    # images: tensor with shape [batch_szie, image_size, image_size, channels]
    # labels: tensor with shape [batch_size, 2 * num_keypoint]
    training_images, training_labels = build_input(FLAGS.train_data_path, batch_size=FLAGS.batch_size, mode=FLAGS.mode)
    print('training_images:', training_images.shape)
    print('training_labels:', training_labels.shape)
    training_model = keypoint_resnet_model.ResNet(hps, training_images, training_labels, mode = 'train')
    training_model.build_graph()

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    training_predictions = training_model.predictions
    training_rmse = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(training_predictions - training_model.labels), axis=1)))

    training_rmse_summary = tf.summary.scalar('training_rmse', training_rmse)

    all_summary = tf.summary.merge_all()

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=FLAGS.train_dir,
        summary_op=all_summary)

    # Prints the given tensors every N local steps, every N seconds, or at end.
    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': training_model.global_step,
                'training_loss': training_model.cost,
                'training_rmse': training_rmse,
                # 'eval_loss': eval_model.cost,
                # 'eval_rmse': eval_rmse,
                },
        every_n_iter=100)

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step.
        Hook to extend calls to MonitoredSession.run().
        """
        # Called once before using the session.
        def begin(self):
            self._lrn_rate = 0.1

        # Called before each call to run().
        def before_run(self, run_context):
            # Represents arguments to be added to a Session.run() call.
            return tf.train.SessionRunArgs(
                    training_model.global_step,  # Asks for global step value.
                    feed_dict={training_model.lrn_rate: self._lrn_rate})  # Sets learning rate

        # Called after each call to run().
        def after_run(self, run_context, run_values):
            """The run_values argument contains results of requested ops/tensors by before_run().
            """
            train_step = run_values.results
            if train_step < 40000:
                self._lrn_rate = 0.1
            elif train_step < 60000:
                self._lrn_rate = 0.01
            elif train_step < 80000:
                self._lrn_rate = 0.001
            else:
                self._lrn_rate = 0.0001

    
    # restore from other platform(operating system), the checkpoint file path in checkpoint file should be changed
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.log_root,
        hooks=[logging_hook, _LearningRateSetterHook()],
        chief_only_hooks=[summary_hook],
        save_summaries_steps=0,
        config=tf.ConfigProto(allow_soft_placement=True)
        ) as mon_sess:
        while not mon_sess.should_stop():
        # for _ in range(700):
            mon_sess.run(training_model.train_op)

def evaluate(hps):
    # Eval the model
    eval_images, eval_labels = build_input(FLAGS.eval_data_path, batch_size=FLAGS.batch_size, mode=FLAGS.mode)
    print('eval_images:', eval_images.shape)
    print('eval_labels:', eval_labels.shape)
    eval_model = keypoint_resnet_model.ResNet(hps, eval_images, eval_labels, mode=FLAGS.mode)
    eval_model.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    eval_predictions = eval_model.predictions
    eval_rmse = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(eval_predictions - eval_model.labels), axis=1)))
    eval_rmse_summary = tf.summary.scalar('eval_rmse', eval_rmse)

    pre_coordinates = (eval_predictions + 1) * FLAGS.image_size / 2
    label_coordinates = (eval_model.labels + 1) * FLAGS.image_size / 2
    coordinates_rmse = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(pre_coordinates - label_coordinates), axis=1)))
    coordinates_rmse_summary = tf.summary.scalar('coordinates_rmse', coordinates_rmse)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    all_summaries = tf.summary.merge([eval_rmse_summary, coordinates_rmse_summary, eval_model.summaries])

    while True:
        try:
            # Returns CheckpointState proto from the "checkpoint" file.
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            continue

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_loss = 0.0
        total_eval_rmse = 0.0
        total_coordinates_rmse = 0.0
        for _ in range(FLAGS.eval_batch_count):
            (all_summaries_str, loss, eval_rmse_value, coordinates_rmse_value, train_step) = sess.run(
                [all_summaries, eval_model.cost, eval_rmse, coordinates_rmse, eval_model.global_step])
            print('loss:', loss, 'eval_rmse:', eval_rmse_value, 
                    'coordinates_rmse:', coordinates_rmse_value)
            total_loss += loss
            total_eval_rmse += eval_rmse_value
            total_coordinates_rmse += coordinates_rmse_value

        average_loss = total_loss / FLAGS.eval_batch_count
        average_eval_rmse = total_eval_rmse / FLAGS.eval_batch_count
        average_coordinates_rmse = total_coordinates_rmse / FLAGS.eval_batch_count

        summary_writer.add_summary(all_summaries_str, train_step)
        print('average_loss:', average_loss, 'average_eval_rmse:', average_eval_rmse, 
            'average_coordinates_rmse:', average_coordinates_rmse)
        summary_writer.flush()
        if FLAGS.eval_once:
            tf.logging.info('Eval Finished!')
            break


def keypoint_detection(model, sess, images, original_image_size=None):
    '''
    demo for keypoint detection
    Params:
        model: network model with graph built.
        sess: tf.Session with model build in
        images: images batch, numpy array with shape [batch_size, height, width, depth].
        label_coords: optimal, unnormalized coordinates labels batch corresponding to the images batch, 
                        numpy array with shape [batch_size, num_keypoint, 2].
        original_image_size: if not provided(None), use the resized image size
    Returns:
        converted_coords: coordinates(x,y) in images, numpy arry with shape [batch_size, num_keypoint, 2]
    '''

    predicted_coords = sess.run(model.predictions, feed_dict={model._images: images})

    if original_image_size is None:
        original_image_size = {'image_width': images.shape[2], 'image_height': images.shape[1]}
    converted_coords = convert_normalized_coordinates(predicted_coords, original_image_size, is_centrolized=False)
    return converted_coords

def test(hps, filename, labels_exist=True):
    '''test csv image data
    '''
    # images:[batch_size,height,eidth, depth]
    images, labels = test_input(filename=filename, labels_exist=labels_exist) 

    normalized_images = (images - np.mean(images, axis=(1,2), keepdims=True)) / np.std(images,axis=(1,2), keepdims=True)
    model, sess = build_test_model(hps)
    test_batch_size = 16

    normalized_images_btatch = normalized_images[:7]
    all_pred_coords = keypoint_detection(model, sess, normalized_images_btatch)
    # all_pred_coords = np.concatenate(all_pred_coords, pred_coords)

    for batch_count in range(111):
        normalized_images_btatch = normalized_images[batch_count*test_batch_size:(batch_count+1)*test_batch_size]
        pred_coords = keypoint_detection(model, sess, normalized_images_btatch)
        all_pred_coords = np.concatenate((all_pred_coords, pred_coords))
        print('batch_conut:', batch_count)


    assert normalized_images.shape[0] == all_pred_coords.shape[0]

    # np.save('test_predict.npy', all_pred_coords)


        # for i in range(test_batch_size):
        #     global_num = batch_count*test_batch_size + i
        #     image = images[global_num]
        #     pred_coord = pred_coords[i]
        #     if labels is not None:
        #         true_label = labels[global_num]
        #     else:
        #         true_label = None

        #     image = image.reshape((image.shape[0],image.shape[1]))
        #     pil_img = Image.fromarray(np.uint8(image),'L')
        #     visualize_keypoints(pil_img, pred_coord, true_keypoints=true_label)
        #     plt.savefig('model_80000_result/' + 'model_80000_detected_test_set_' + str(global_num) + '.jpg')
        #     plt.show()
            


def demo(image_filenames, hps, use_original_img=False):
    '''
    detect each image store in image_filenames and disply it 
    Params:
        image_filenames: a list of filenames
    '''

    model, sess = build_test_model(hps)
    for filename in image_filenames:
        img = Image.open(filename)

        mono_image, resized_image = preprocess_image(img)

        mono_image = np.float32(mono_image)
        mono_image = mono_image.reshape((1, mono_image.shape[0], mono_image.shape[1], 1))
        
        if use_original_img:
            original_image_size = {'image_width':img.width, 'image_height':img.height}
        else:
            original_image_size =None
        pred_coord = keypoint_detection(model, sess, mono_image, original_image_size)
        print('------'*10)
        print('detection result for:', filename)
        print('predicted keypoints:\n',pred_coord)
        if use_original_img:
            visualize_keypoints(img, pred_coord, None)
        else:
            visualize_keypoints(resized_image, pred_coord, None)
        plt.show()

def build_test_model(hps):
    model = keypoint_resnet_model.ResNet(hps, mode='test')
    model.build_graph()
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    try:
        # Returns CheckpointState proto from the "checkpoint" file.
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', FLAGS.log_root)

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)


    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    return model, sess


def visualize_keypoints(image, pre_keypoints, true_keypoints=None):
    '''plot predicted and true keypoints (if not None) on image
    Params:
        pre_keypoints: coordinates of keypoint with shape [num_keypoint, 2], each row [x, y]
    '''

    if image.mode == 'L':
        plt.imshow(image, cmap='gray') # cmap='gray'
    else:
        plt.imshow(image)

    if pre_keypoints is not None:
        plt.scatter(pre_keypoints[:,:,0], pre_keypoints[:,:,1])
    if true_keypoints is not None:
        plt.scatter(true_keypoints[:,:,0], true_keypoints[:,:,1], c='r', marker='x')

    # plt.savefig('model80000_detected_' + str(counter) + '.jpg')


def preprocess_image(image):
    '''process imgae to be which resnet can receieve
    '''

    # convert image to gray scale
    if image.mode != 'L':
        image = image.convert('L')

    # resize image to FLAGS.image_size * FLAGS.image_size
    resized_image = image.resize((FLAGS.image_size, FLAGS.image_size))

    # standrdlize image
    image = (resized_image - np.mean(resized_image)) / np.std(resized_image)

    return image, resized_image

def convert_normalized_coordinates(normalized_coordinates, original_image_size, is_centrolized=False):
    '''convert normalize_coordinates to coordinates in original image,
    Params:
        normalized_coordinates: predicted coordinates from resnet, with shape [batch_size, 2 * num_keypoint]
        original_image_size: a dict {'image_width': image_width, 'image_height': image_height} 
                            represents the size of original image(before resize)
        is_centrolized: True, if normalized coordinates range (-1, 1), Flase, range(0, 1)
    Return:
        converted_coordinates: coordinates of keypoint with shape [batch_size, num_keypoint, 2], each row [x, y]
    '''

    converted_coordinates = normalized_coordinates.reshape((normalized_coordinates.shape[0], FLAGS.num_keypoint, 2))

    if is_centrolized:
        converted_coordinates[:,:,0] = (converted_coordinates[:,:,0] + 1) * original_image_size['image_width'] / 2 # x coords
        converted_coordinates[:,:,1] = (converted_coordinates[:,:,1] + 1) * original_image_size['image_height'] / 2 # y coords
    else:
        converted_coordinates[:,:,0] = converted_coordinates[:,:,0] * original_image_size['image_width'] # x coords
        converted_coordinates[:,:,1] = converted_coordinates[:,:,1] * original_image_size['image_height'] # y coords

    converted_coordinates[:,:,0] = converted_coordinates[:,:,0].clip(0,original_image_size['image_width'])
    converted_coordinates[:,:,1] = converted_coordinates[:,:,1].clip(0,original_image_size['image_height'])
    
    return converted_coordinates


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')    


    hps = keypoint_resnet_model.HParams(
                                # batch_size=FLAGS.batch_size,
                                num_keypoint=FLAGS.num_keypoint,
                                min_lrn_rate=0.0001,
                                lrn_rate=0.1,
                                num_residual_units=5,
                                use_bottleneck=False,
                                weight_decay_rate=0.0002,
                                relu_leakiness=0.1,
                                optimizer='mom')


    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
        elif FLAGS.mode == 'eval':
            evaluate(hps)
        elif FLAGS.mode == 'test':
            # img = Image.open(FLAGS.demo_image_path)
            # img = img.resize((FLAGS.image_size, FLAGS.image_size))
            # img.show('ii')


            # imags, labels = test_input(filename=FLAGS.eval_data_path, labels_exist=True)
            # img = imags[counter].reshape((FLAGS.image_size, FLAGS.image_size))
            # img = Image.fromarray(np.uint8(img),'L')
            # label = labels[counter]
            # demo(hps, img, label_coords=label)
            # plt.show()

            test(hps, filename = FLAGS.eval_data_path, labels_exist=True)
        else:
            image_filenames = ['demo_images/demo_4.jpg', 'demo_images/demo_2.png','demo_images/demo_3.png',]
            demo(image_filenames, hps, use_original_img = False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()        
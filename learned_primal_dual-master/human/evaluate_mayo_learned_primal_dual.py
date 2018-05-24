import os
import adler
import pydicom as dicom
adler.util.gpu.setup_one_gpu()

from adler.tensorflow import prelu, psnr
from scipy.misc import imsave

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
from mayo_util import DATA_FOLDER, FileLoader
import cv2

np.random.seed(0)
name = 'mayo_learned_primal_dual'

sess = tf.InteractiveSession()
# print('Checkpoints saved in directory: ' +
                # adler.tensorflow.util.default_checkpoint_path(name))

# Create ODL data structures
size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size],
                          dtype='float32', weighting='const')

# Tomography
# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 912)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-360, 360, 912)
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    # src_radius=500, det_radius=500)
                                    src_radius=538.520000, det_radius=408.226000)


operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

# For some reason adjoint operator didn't work, will report 'loss = nan', otherwise we would use operator.adjoint
# instead of pseudoinverse
# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator,
                                                          'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse,
                                                                  'RayTransformAdjoint')

# User selected paramters
n_data = 1
n_iter = 10
n_primal = 5
n_dual = 5
mu_water = 0.02
photons_per_pixel = 10000
DATA_FOLDER = '/media/tx-eva-cc/data/2018_03_26_WuHanTongJi/1517110864/LDHDLungSS40'
dcm_name = '1517110864_075.dcm'
#'/media/tx-eva-cc/data/2018_03_26_WuHanTongJi/1517110864/LDHDLungSS40'

# '/media/tx-eva-cc/data/WHTJ_Test/1517068707'

file_loader = FileLoader(DATA_FOLDER, exclude='L286')


def generate_data(validation=False, data_folder=DATA_FOLDER):
    """Generate a set of random data."""
    # n_iter = 1 if validation else n_data

    # y_arr = np.empty((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    # x_true_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    # if validation:
    #     n_data = len([fname for fname in os.listdir(validation_path)
    #                   if os.path.isfile(os.path.join(validation_path, fname))])
    # else:
    #     n_data = len([fname for fname in os.listdir(train_path)
    #                   if os.path.isfile(os.path.join(train_path, fname))])
    # y_arr = np.empty((n_data, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((1, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr = np.empty((1, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')

    fi = os.path.join(data_folder, dcm_name)
    dicom_file = dicom.read_file(fi)
    np_data = dicom_file.pixel_array.astype('float32') * dicom_file.RescaleSlope \
                    + dicom_file.RescaleIntercept
    np_data[np_data < -1024.0] = -1024.0
    np_data[np_data > 3071.0] = 3071.0
    np_data += 1024.0
    print(np_data.max()-np_data.min())

    phantom = space.element(np.rot90(np_data, -1))
    phantom /= 1000.0  # convert go g/cm^3

    data = operator(phantom)
    data = np.exp(-data * mu_water)

    # turn off Poisson noise for reconstruction
    # noisy_data = odl.phantom.poisson_noise(data * photons_per_pixel) / photons_per_pixel

    x_true_arr[0, ..., 0] = phantom
    y_arr[0, ..., 0] = data

    # return y_arr
    return x_true_arr, y_arr


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true")
    y_rt = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], 1], name="y_rt")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')


def apply_conv(x, filters=32):
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

primal_values = []
dual_values = []

with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.concat([tf.zeros_like(x_true)] * n_primal, axis=-1)
        dual = tf.concat([tf.zeros_like(y_rt)] * n_dual, axis=-1)

    for i in range(n_iter):
        with tf.variable_scope('dual_iterate_{}'.format(i)):
            evalpt = primal[..., 1:2]
            evalop = tf.maximum(tf.exp(-mu_water * odl_op_layer(evalpt)), tf.exp(-10.0))
            update = tf.concat([dual, evalop, y_rt], axis=-1)

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_dual)
            dual = dual + update

        with tf.variable_scope('primal_iterate_{}'.format(i)):
            evalpt_fwd = primal[..., 0:1]
            evalop_fwd = (-mu_water) * tf.exp(-mu_water * odl_op_layer(evalpt_fwd))

            evalpt = dual[..., 0:1]
            evalop = odl_op_layer_adjoint(evalop_fwd * dual[..., 0:1])
            update = tf.concat([primal, evalop], axis=-1)

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_primal)
            primal = primal + update

        primal_values.append(primal)
        dual_values.append(dual)

    x_result = primal[..., 0:1]

with tf.name_scope('loss'):
    residual = x_result - x_true
    squared_error = residual ** 2
    loss = tf.reduce_mean(squared_error)

# def loss_L2(image_1, image_2):
#     residual = image_1 - image_2
#     squared_error = residual ** 2
#     loss = tf.reduce_mean(squared_error)
#     return loss

# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

if 1:
    saver.restore(sess,
                  # adler.tensorflow.util.default_checkpoint_path(name))
                  "/mnt/data2/inverse_problems/ct_reconstruction/learned_primal_dual/human/model/test_pseudoinv_adam--16.ckpt")
# Generate validation data
x_true_arr_validate, y_arr_validate = generate_data(validation=True)

loss_result, primal_values_result, dual_values_result = sess.run([loss, primal_values, dual_values],
                      feed_dict={x_true: x_true_arr_validate,
                                 y_rt: y_arr_validate,
                                 is_training: False})
print ('loss={}'.format(loss_result))
# print (type(primal_values_result[-1][..., 0:1]))

import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


# jpg_img = (((img - np.min(img))/4.095) * 255).astype('uint8')
# print jpg_img


# cv2.imwrite('/media/tx-eva-cc/data/WHTJ_Test/reconstructed_images/1517110864/cv.jpg',
# cv2.imread('/media/tx-eva-cc/data/WHTJ_Test/reconstructed_images/1517110864/img.jpg'))
# np.save('/media/tx-eva-cc/data/WHTJ_Test/reconstructed_images/1517110864/image.npy', primal_values_result[-1][..., 0:1]-1.024)
# print (np.histogram(np.load('/media/tx-eva-cc/data/WHTJ_Test/reconstructed_images/1517110864/image.npy')))

# print (np.max(x_true_arr_validate[..., 0:1]), np.min(x_true_arr_validate[..., 0:1]))
# print (np.histogram(x_true_arr_validate[..., 0:1]))
# cv2.imshow('image', cv2.imread('/media/tx-eva-cc/data/WHTJ_Test/reconstructed_images/1517110864/image.npy'))

# Compare inverse-engineered image with its counterpart corresponding to a specific reconstruction algorithm
x_compare, _ = generate_data(validation=True,
                        data_folder='/media/tx-eva-cc/data/2018_03_26_WuHanTongJi/1517110864/LDStdSS0')
# print (np.max(x_compare[..., 0:1]), np.min(x_compare[..., 0:1]))
# print (np.histogram(x_compare[..., 0:1]))
print('ssim(train_image, test_image): %f' %ssim(x_true_arr_validate[0, ..., 0], x_compare[0, ..., 0]))
print('loss_L2(train_image, test_image): %f' %sess.run([loss],
                                                       feed_dict={x_result: x_true_arr_validate[..., 0:1],
                                                                  x_true: x_compare[..., 0:1]})[0])
print('ssim(reconstructed_image, train_image): %f' %ssim(primal_values_result[-1][0, ..., 0], x_true_arr_validate[0, ..., 0]))
print('loss_L2(reconstructed_image, train_image): %f' %sess.run([loss],
                                                       feed_dict={x_result: primal_values_result[-1][..., 0:1],
                                                                  x_true: x_true_arr_validate[..., 0:1]})[0])
print('ssim(reconstructed_image, test_image): %f'  %ssim(primal_values_result[-1][0, ..., 0], x_compare[0, ..., 0]))
print('loss_L2(reconstructed_image, test_image): %f' %sess.run([loss],
                                                       feed_dict={x_result: primal_values_result[-1][..., 0:1],
                                                                  x_true: x_compare[..., 0:1]})[0])

print(psnr(primal_values_result[-1][0, ..., 0], x_true_arr_validate[0, ..., 0], dynamic_range=np.max(x_true_arr_validate) - np.min(x_true_arr_validate)))



def normalized(val, sign=False):
    if sign:
        val = val * np.sign(np.mean(val))
    return (val - np.mean(val)) / np.std(val)

# print (np.shape(img))

path = os.path.join('/mnt/data2/inverse_problems/ct_reconstruction/learned_primal_dual/data/evaluation/human', name)
if not os.path.exists(path):
    os.mkdir(path)

for i in range(n_iter):
    vals = primal_values_result[i]
    space.element(vals[..., 0]).show(saveto='{}/x_{}'.format(path, i),cmap = 'gray')
    space.element(vals[..., 0]).show(clim=[0.8, 1.2], saveto='{}/x_windowed_{}'.format(path, i),cmap = 'gray')
    space.element(normalized(primal_values_result[i][..., 1], True)).show(clim=[-3, 3], saveto='{}/x_eval_{}'.format(path, i),cmap = 'gray')
    operator.range.element(normalized(dual_values_result[i][..., 0])).show(clim=[-3, 3], saveto='{}/y_{}'.format(path, i),cmap = 'gray')

    plt.close('all')

el = space.element(primal_values_result[-1][..., 0])
el.show('', coords=[[-40, 25], [-25, 25]], clim=[0.8, 1.2], saveto='{}/x_midle'.format(path), cmap = 'gray')

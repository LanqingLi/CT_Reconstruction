import os
import adler
import pydicom as dicom
adler.util.gpu.setup_one_gpu()

from adler.tensorflow import prelu, cosine_decay, psnr

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
from mayo_util import FileLoader, DATA_FOLDER

np.random.seed(0)
name = os.path.splitext(os.path.basename(__file__))[0]

sess = tf.InteractiveSession()
# print('Checkpoints being saved in directory: ' +
                # adler.tensorflow.util.default_checkpoint_path(name))

# Create ODL data structures
size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size],
                          dtype='float32', weighting='const')

# Tomography
# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 500)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-360, 360, 500)
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    # src_radius=500, det_radius=500)
                                    src_radius=538.520000, det_radius=408.226000)


operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)
# For some reason adjoint operator didn't work, will report 'loss = nan', otherwise we would use operator.adjoint
# instead of pseudoinverse
# operator_adjoint = odl.tomo.RayBackProjection(space, geometry)

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator,
                                                          'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse,
                                                                  'RayTransformAdjoint')

# User selected paramters
n_iter = 10
n_primal = 5
n_dual = 5
mu_water = 0.02
photons_per_pixel = 10000
batch_size = 1
DATA_FOLDER = '/media/tx-eva-cc/data/WHTJ_Test/1517068707'
train_path = os.path.join(DATA_FOLDER, 'train')
validation_path = os.path.join(DATA_FOLDER, 'validation')
# print  [os.path.isfile(os.path.join(validation_path,fname)) for fname in os.listdir(validation_path)]

file_loader_train = FileLoader(train_path, exclude='')
file_loader_validation = FileLoader(validation_path, exclude='')

def generate_data(validation=False):
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
    x_true_arr = np.empty((batch_size, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr = np.empty((batch_size, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')

    # if validation:
    #     fi = os.path.join(validation_path, os.listdir(validation_path)[img_num])
    # else:
    #     fi = os.path.join(train_path, os.listdir(train_path)[img_num])
            # print 'Adding training data: %s' %fi
    if validation:
        fi = file_loader_validation.next_file()
    else:
        fi = file_loader_train.next_file()
    dicom_file = dicom.read_file(fi)
    np_data = dicom_file.pixel_array.astype('float32') * dicom_file.RescaleSlope \
                    + dicom_file.RescaleIntercept
    np_data[np_data < -1024.0] = -1024.0
    np_data[np_data > 3071.0] = 3071.0
    np_data += 1024.0

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
    return tf.layers.conv2d(x, filters=filters, kernel_size=(3, 3), padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())


with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.concat([tf.zeros_like(x_true)] * n_primal, axis=-1)
        dual = tf.concat([tf.zeros_like(y_rt)] * n_dual, axis=-1)

    for i in range(n_iter):
        with tf.variable_scope('dual_iterate_{}'.format(i)):
            evalpt = primal[..., 1:2]
            # prevent overflow when attenuation is large
            evalop = tf.maximum(tf.exp(-mu_water * odl_op_layer(evalpt)), tf.exp(-10.0))
            update = tf.concat([dual, evalop, y_rt], axis=-1)
            # print np.shape(update)
            update = prelu(apply_conv(update), name='prelu_1')
            # y_arr, x_true_arr = generate_data()
            # print (sess.run(update, feed_dict={x_true: x_true_arr,
            #                              y_rt: y_arr}))
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_dual)
            dual = dual + update

        with tf.variable_scope('primal_iterate_{}'.format(i)):
            evalpt_fwd = primal[..., 0:1]
            evalop_fwd = (-mu_water) * tf.exp(-mu_water * odl_op_layer(evalpt_fwd))

            evalpt = dual[..., 0:1]
            evalop = odl_op_layer_adjoint(evalop_fwd * evalpt)
            update = tf.concat([primal, evalop], axis=-1)

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_primal)
            primal = primal + update

    x_result = primal[..., 0:1]


with tf.name_scope('loss'):
    residual = x_result - x_true
    squared_error = residual ** 2
    loss = tf.reduce_mean(squared_error)



with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    maximum_steps = 100
    starter_learning_rate = 1e-5
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')
    #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                       10000, 0.5, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                beta2 = 0.999, epsilon=1e-08)
        # opt_func = tf.train.MomentumOptimizer(learning_rate=learning_rate,
        #                                        momentum=0.9,use_nesterov=True)
        # train = opt_func.minimize(loss)


        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        optimizer = opt_func.apply_gradients(zip(grads, tvars),
                                             global_step=global_step)


# Summaries
# tensorboard --logdir=...

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('psnr', psnr(x_result, x_true, 'compute_psnr'))

    tf.summary.image('x_result', x_result)
    tf.summary.image('x_true', x_true)
    tf.summary.image('squared_error', squared_error)
    tf.summary.image('residual', residual)

    merged_summary = tf.summary.merge_all()
    test_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/test',
                                                sess.graph)
    train_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/train')

# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

# Generate validation data
#y_arr_validate
# x_true_arr_validate, y_arr_validate = generate_data(validation=True)

if 0:
    saver.restore(sess,
                  #adler.tensorflow.util.default_checkpoint_path(name))
                  '/mnt/data2/inverse_problems/ct_reconstruction/learned_primal_dual/data/model/test_pseudoinv_adam--16.ckpt')
# Train the network
for i in range(0, maximum_steps):
    n_data = len([fname for fname in os.listdir(train_path)
                      if os.path.isfile(os.path.join(train_path, fname))])
    for img_num in range(n_data):

        # if i%10 == 0:
        x_true_arr, y_arr = generate_data(validation=False)
        # print(tf.reduce_mean(y_arr))
        # print sess.run([optimizer, merged_summary, global_step],
        #                           feed_dict={x_true: x_true_arr,
        #                                      y_rt: y_arr,
        #                                      is_training: True})
        loss_result, _, merged_summary_result_train, global_step_result = sess.run([loss, optimizer, merged_summary, global_step],
                                  feed_dict={x_true: x_true_arr,
                                             y_rt: y_arr,
                                             is_training: True})
        # print (sess.run(y_rt, feed_dict={y_rt: y_arr}))
        # print (merged_summary_result_train)
        # print (opt)
        # print (tf.shape(x_true_arr))
        # if i>0 and i%10 == 0:
        # loss_result, merged_summary_result, global_step_result = sess.run([loss, merged_summary, global_step],
        #                           feed_dict={x_true: x_true_arr_validate,
        #                                      y_rt: y_arr_validate,
        #                                      is_training: False})

        train_summary_writer.add_summary(merged_summary_result_train, global_step_result)
        # test_summary_writer.add_summary(merged_summary_result, global_step_result)

        print('iter={}, img_num = {}, loss={}'.format(i, img_num, loss_result))

        if img_num == n_data - 1:
        # saver.save(sess,
        #           adler.tensorflow.util.default_checkpoint_path(name))
            saver.save(sess,
                       "/mnt/data2/inverse_problems/ct_reconstruction/learned_primal_dual/data/model/test_pseudoinv_adam_1set--%i.ckpt" %(i))
        # saver.save(sess, 'tf_variables')
        # if global_step_result > maximum_steps:
        #     break

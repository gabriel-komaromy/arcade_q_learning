import ale_python_interface.ale_python_interface as ale
import tensorflow as tf
from scipy import misc
import numpy as np

breakout = ale.ALEInterface()
path = '/Users/Gabe/classes/current/600/project/Arcade-Learning-Environment' + \
    '-0.5.1/roms/Breakout.bin'

breakout.loadROM(path)
# breakout.getLegalActionSet()

"""This is the one I probably want to use, it's only the actions that have
an effect in the game"""
# breakout.getMinimalActionSet()
# breakout.getScreen()

"""Outputs 210 x 160 image. You can also pass in an np array if you want it to
fill that for you. Actually, that produces weirdly high inputs, don't fuck with
it."""
# grays = breakout.getScreenGrayscale()

"""It has shape (210, 160, 1) and we need (210, 160)"""
# grays_2d = grays.reshape((210, 160))
# downsampled_grays = misc.imresize(grays_2d, (108, 84))

"""Saves a PNG output of the screen, will probably be useful for report"""
# breakout.saveScreenPNG('first_frame.png')


def frame_to_q_input(grayscale_frame):
    frame_2d = grayscale_frame.reshape(grayscale_frame.shape[:2])
    downsampled = misc.imresize(frame_2d, (108, 84))
    return downsampled


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d(x, W, stride_size):
    return tf.nn.conv3d(
        x,
        W,
        strides=[1, 1, stride_size, stride_size, 1],
        padding='SAME',
        )


def new_frame():
    breakout.act(1)
    grayscale_frame = breakout.getScreenGrayscale()
    downsampled = frame_to_q_input(grayscale_frame)
    return downsampled


def frame_stack():
    frames_array = np.zeros([4, 108, 84])
    for i in xrange(4):
        frames_array[i] = new_frame()

    tf_frames = tf.reshape(frames_array, [-1, 4, 108, 84, 1])
    float_frames = tf.to_float(tf_frames)
    return float_frames


trial_stack = frame_stack()
W_conv1 = weight_variable([1, 8, 8, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv3d(trial_stack, W_conv1, 4) + b_conv1)

W_conv2 = weight_variable([1, 4, 4, 16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv3d(h_conv1, W_conv2, 2) + b_conv2)
h_conv2_flat = tf.reshape(h_conv2, [-1, 4 * 14 * 11 * 32])

W_fc1 = weight_variable([4 * 14 * 11 * 32, 256])
b_fc1 = bias_variable([256])

h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

num_actions = 4

W_fc2 = weight_variable([256, num_actions])
b_fc2 = bias_variable([num_actions])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

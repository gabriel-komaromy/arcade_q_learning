import ale_python_interface.ale_python_interface as ale
import tensorflow as tf
from scipy import misc
import numpy as np

breakout = ale.ALEInterface()
breakout.loadROM('../Arcade-Learning-Environment-0.5.1/roms/Breakout.bin')
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

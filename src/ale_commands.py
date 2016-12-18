import ale_python_interface.ale_python_interface as ale
import tensorflow as tf
from scipy import misc
import numpy as np
import random

"""I'm following the lead of
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html,
although the problem here requires significant alterations."""

breakout = ale.ALEInterface()
path = '/Users/Gabe/classes/current/600/project/Arcade-Learning-Environment' + \
    '-0.5.1/roms/Breakout.bin'

breakout.loadROM(path)

available_actions = breakout.getMinimalActionSet()
num_actions = len(available_actions)

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

# from Nature paper
downsampled_w = 108
downsampled_h = 84
stack_size = 4
batch_size = 32

epsilon = 0.5


def frame_to_q_input(grayscale_frame):
    frame_2d = grayscale_frame.reshape(grayscale_frame.shape[:2])
    downsampled = misc.imresize(frame_2d, (downsampled_w, downsampled_h))
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


def new_frame(action):
    reward = breakout.act(action)
    grayscale_frame = breakout.getScreenGrayscale()
    downsampled = frame_to_q_input(grayscale_frame)
    return downsampled, reward


def frame_stack(action):
    assert action in xrange(num_actions)
    ale_action = available_actions[action]
    frames_array = np.zeros([stack_size, downsampled_w, downsampled_h])
    total_reward = 0
    for i in xrange(stack_size):
        frames_array[i], current_reward = new_frame(ale_action)
        total_reward += current_reward

    reshaped_frames = np.reshape(
        frames_array,
        [-1, stack_size, downsampled_w, downsampled_h, 1],
        )
    float_frames = reshaped_frames.astype(np.float32)
    average_reward = total_reward / float(stack_size)
    return float_frames, average_reward


def inference(inputs):
    # batch_size = tf.shape(inputs[0])[0]
    conv1_units = 16
    conv2_units = 32
    fully_connected_units = 128
    W_conv1 = weight_variable([1, 8, 8, 1, conv1_units])
    b_conv1 = bias_variable([conv1_units])
    h_conv1 = tf.nn.relu(conv3d(inputs, W_conv1, 4) + b_conv1)

    W_conv2 = weight_variable([1, 4, 4, conv1_units, conv2_units])
    b_conv2 = bias_variable([conv2_units])

    conv2_outputs = stack_size * 14 * 11 * conv2_units
    h_conv2 = tf.nn.relu(conv3d(h_conv1, W_conv2, 2) + b_conv2)
    h_conv2_flat = tf.reshape(h_conv2, [-1, conv2_outputs])

    W_fc1 = weight_variable([conv2_outputs, fully_connected_units])
    b_fc1 = bias_variable([fully_connected_units])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([fully_connected_units, num_actions])
    b_fc2 = bias_variable([num_actions])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv


def batch_loss(taken_q, max_q, rewards, discount_factor):
    discounted_q = tf.scalar_mul(discount_factor, taken_q)
    discounted_q = tf.Print(discounted_q, [discounted_q], 'discounted q')
    target = tf.add(discounted_q, rewards)
    target = tf.Print(target, [target], 'target')
    difference = tf.subtract(target, max_q)
    difference = tf.Print(difference, [difference], 'difference')
    loss = tf.square(difference)
    loss = tf.Print(loss, [loss], 'loss')
    return loss


def train(loss, learning_rate):
    # from Nature paper
    momentum = 0.95
    optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # global_step = tf.Print(global_step, [global_step], 'global_step')
    train_op = optimizer.minimize(loss, global_step=global_step)
    # train_op = tf.Print(train_op, [train_op], 'train op')
    return train_op


def fill_feed_dict(
        last_frames_placeholder,
        next_frames_placeholder,
        rewards_placeholder,
        action_placeholder,
        current_step,
        memory,
        ):
    samples = random.sample(memory, batch_size)
    trimmed_shape = [stack_size, downsampled_w, downsampled_h, 1]
    last_frames = [
        transition[0].reshape(trimmed_shape) for transition in samples
        ]
    next_frames = [
        transition[1].reshape(trimmed_shape) for transition in samples
        ]
    actions = np.reshape(
        [transition[2] for transition in samples],
        (batch_size, 1),
        )
    rewards = [transition[3] for transition in samples]
    feed_dict = {
        last_frames_placeholder: last_frames,
        next_frames_placeholder: next_frames,
        action_placeholder: actions,
        rewards_placeholder: rewards,
        }

    return feed_dict


def get_best_action(last_frame_stack):
    all_q_values = inference(last_frame_stack)
    """This isn't ideal but I'm not sure how to make tf give me a value rather
    than a tensor."""
    best_action = np.argmax(all_q_values)
    if random.random() < epsilon:
        action = random_action()
    else:
        action = best_action
    return action


def random_action():
    return random.choice(range(num_actions))


def run_training():
    with tf.Graph().as_default():
        # from Nature paper
        discount_factor = 0.99
        learning_rate = 0.00025

        # stuff I made up
        max_steps = 100
        memory_size = 100

        """We want to be able to input either a full batch, for training, or a
        single frame stack, for getting the next Q-value. The None in shape
        allows us to do either."""
        current_frames = tf.placeholder(
            tf.float32,
            shape=[None, stack_size, downsampled_w, downsampled_h, 1],
            )

        next_frames = tf.placeholder(
            tf.float32,
            shape=[None, stack_size, downsampled_w, downsampled_h, 1],
            )

        rewards = tf.placeholder(
            tf.float32,
            shape=[None],
            )

        """Basically, current_q needs to incorporate the action that was taken,
        so it calls inference but then indexes into it. next_q needs to call
        inference and then find the max action from that."""
        current_q = inference(current_frames)
        # print 'current_q', current_q
        action_taken_index = tf.placeholder(tf.int32, shape=[32, 1])
        indexes = [[n, 0] for n in xrange(batch_size)]
        actions_taken = tf.gather_nd(action_taken_index, indexes)
        """
        actions_taken = tf.Print(
            actions_taken,
            [actions_taken],
            'actions tak',
            )
            """
        index_range = tf.range(batch_size)
        coords = tf.transpose(tf.pack([index_range, actions_taken]))
        # coords = tf.Print(coords, [coords], 'coords', summarize=5)
        # action_locations = [[index, action] for action in actions_taken]
        # taken_q = current_q[action_taken_index]
        taken_q = tf.gather_nd(current_q, coords)
        # taken_q = tf.Print(taken_q, [taken_q], 'taken q')
        # print taken_q, 'taken q'

        next_q = inference(next_frames)
        max_q = tf.reduce_max(next_q)

        loss = batch_loss(taken_q, max_q, rewards, discount_factor)

        train_op = train(loss, learning_rate)

        init = tf.global_variables_initializer()

        sess = tf.Session()

        sess.run(init)

        initial_action = random_action()

        last_frame_stack, average_reward = frame_stack(initial_action)

        memory = [None] * memory_size

        for starting_step in xrange(batch_size):
            print 'starting step', starting_step
            action_choice = get_best_action(last_frame_stack)
            new_frame_stack, average_reward = frame_stack(action_choice)
            adding_tuple = (
                last_frame_stack,
                new_frame_stack,
                action_choice,
                average_reward
                )

            memory[starting_step % memory_size] = adding_tuple
            last_frame_stack = new_frame_stack

        print 'past initial memory fill'

        for step in xrange(batch_size, max_steps):
            print 'starting step', step
            action_choice = get_best_action(last_frame_stack)
            print action_choice, 'action_choice'
            new_frame_stack, average_reward = frame_stack(action_choice)

            memory[step % memory_size] = (
                last_frame_stack,
                new_frame_stack,
                action_choice,
                average_reward
                )

            if step < memory_size:
                filled_memory = memory[:step]
            else:
                filled_memory = memory

            feed_dict = fill_feed_dict(
                current_frames,
                next_frames,
                rewards,
                action_taken_index,
                step,
                filled_memory,
                )

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            last_frame_stack = new_frame_stack

            loss_value = loss_value
run_training()

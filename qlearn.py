from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
import sys

if "../" not in sys.path:
    sys.path.append("../")

import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import tensorflow as tf
import itertools

from model import *

GAME = 'bird'  # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 3200.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4  # We stack 4 frames


def buildmodel():
    print("Now we build the model")
    model = simple_model.Simple([img_rows, img_cols, img_channels], ACTIONS, learning_rate=LEARNING_RATE,
                                loss_name='mean_squared_error')
    print("We finish building the model")
    return model


def trainNetwork(model, args):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    x_t = x_t / 255.0

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    # print (s_t.shape)

    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.restore(sess)

        if args['mode'] == 'Run':
            OBSERVE = 999999999  # We keep observe, never train
            epsilon = 0
        else:  # We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON

        t = 0
        while (True):
            episode_length = 0
            episode_reward = 0
            for iter_i in itertools.count():
                loss = 0
                Q_sa = 0
                action_index = 0
                r_t = 0
                a_t = np.zeros([ACTIONS])
                # choose an action epsilon greedy
                if t % FRAME_PER_ACTION == 0:
                    if random.random() <= epsilon:
                        # print("----------Random Action----------")
                        action_index = random.randrange(ACTIONS)
                        a_t[action_index] = 1
                    else:
                        q = model.predict(sess, s_t)  # input a stack of 4 images, get the prediction
                        max_Q = np.argmax(q)
                        action_index = max_Q
                        a_t[max_Q] = 1

                # We reduced the epsilon gradually
                if epsilon > FINAL_EPSILON and t > OBSERVE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                # run the selected action and observed next state and reward
                x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

                episode_length += 1
                episode_reward += r_t

                x_t1 = skimage.color.rgb2gray(x_t1_colored)
                x_t1 = skimage.transform.resize(x_t1, (80, 80))
                x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

                x_t1 = x_t1 / 255.0

                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
                s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

                # store the transition in D
                D.append((s_t, action_index, r_t, s_t1, terminal))
                if len(D) > REPLAY_MEMORY:
                    D.popleft()

                # only train if done observing
                if t > OBSERVE:
                    # sample a minibatch to train on
                    minibatch = random.sample(D, BATCH)

                    # Now we do the experience replay
                    state_t, action_t, reward_t, state_t1, terminal_batch = zip(*minibatch)
                    state_t = np.concatenate(state_t)
                    state_t1 = np.concatenate(state_t1)
                    targets = model.predict(sess, state_t)
                    Q_sa = model.predict(sess, state_t1)
                    targets[range(BATCH), action_t] = reward_t + GAMMA * np.max(Q_sa, axis=1) * np.invert(
                        terminal_batch)

                    loss += model.update(sess, state_t, targets)

                    # save progress every 10000 iterations
                    if t % 1000 == 0:
                        print("Now we save model")
                        model.save(sess)

                s_t = s_t1
                t = t + 1

                # print info
                state = ""
                if t <= OBSERVE:
                    state = "observe"
                elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                    state = "explore"
                else:
                    state = "train"

                if t % 100 == 0:
                    print("TIMESTEP", t, "/ EPISODE_LENGTH", episode_length, "/ STATE", state, \
                          "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                          "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

                if terminal:
                    break

            # Add summaries to tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=episode_reward, node_name="episode_reward",
                                      tag="episode_reward")
            episode_summary.value.add(simple_value=episode_length, node_name="episode_length",
                                      tag="episode_length")
            model.train_writer.add_summary(episode_summary, sess.run(tf.train.get_global_step()))
            model.train_writer.flush()

        model.train_writer.close()
        model.validation_writer.close()
        print("Episode finished!")
        print("************************")


def playGame(args):
    model = buildmodel()
    trainNetwork(model, args)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)


if __name__ == "__main__":
    main()

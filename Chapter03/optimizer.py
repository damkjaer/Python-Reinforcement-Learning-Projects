#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


class Optimizer:

    def __init__(self, config, feedback_size, q_network, target_network, replay_memory):

        self.feedback_size = feedback_size
        self.q_network = q_network
        self.target_network = target_network
        self.replay_memory = replay_memory
        self.summary_writer = None

        self.gamma = config['gamma']
        self.num_frames = config['num_frames']

        if config['optimizer'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=config['learning_rate'],
                decay=config['rho'],
                epsilon=config['rmsprop_epsilon'])
        elif config['optimizer'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=config['learning_rate'],
                momentum=config['rho'])
        elif config['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=config['learning_rate'],
                beta1=config['rho'])
        else:
            raise ValueError('Unknown optimizer')
            
        self.train_op = optimizer.apply_gradients(
            zip(self.q_network.gradient,
                self.q_network.vars))

    def set_summary_writer(self, summary_writer=None):
                self.summary_writer = summary_writer
        
    def sample_transitions(self, sess, batch_size):

        w, h = self.feedback_size
        states = np.zeros((batch_size, self.num_frames, w, h), dtype=np.float32)
        new_states = np.zeros((batch_size, self.num_frames, w, h), dtype=np.float32)
        targets = np.zeros(batch_size, dtype=np.float32)
        actions = np.zeros(batch_size, dtype=np.int32)
        terminations = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            state, action, r, new_state, t = self.replay_memory.sample()
            states[i] = state
            new_states[i] = new_state
            actions[i] = action
            targets[i] = r
            terminations[i] = t

        targets += self.gamma * (1 - terminations) * self.target_network.get_q_value(sess, new_states)
        return states, actions, targets

    def train_one_step(self, sess, step, batch_size):

        states, actions, targets = self.sample_transitions(sess, batch_size)
        feed_dict = self.q_network.get_feed_dict(states, actions, targets)

        if self.summary_writer and step % 1000 == 0:
            summary_str, _, = sess.run([self.q_network.summary_op,
                                        self.train_op],
                                       feed_dict=feed_dict)
            self.summary_writer.add_summary(summary_str, step)
            self.summary_writer.flush()
        else:
            sess.run(self.train_op, feed_dict=feed_dict)


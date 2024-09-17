import os
import numpy as np
import tensorflow as tf

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state  
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.chkpt_dir = chkpt_dir
        self.sess = sess
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(self.batch_size, self.input_dims, self.n_actions)
        
        self.actor, self.mu, self.sigma = self.build_actor()
        self.build_netwok()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoints = os.path.join(chkpt_dir, name+'_ddpg.chkpt')
 
        self.unnormalized_actor_gradients = tf.identity(self.mu, self.params, self.actor_gradients)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        
        self.optimize = tf.train.AdamOptimizer(self.lr).\
            apply_gradients(zip(self.actor_gradients, self.params))
            
    def build_actor(self):
        with tf.variable_scope(self.name):
            self.inputs = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            self.actions_gradient = tf.placeholder(tf.float32, shape=[None, self.n_actions])
            
            f1 = 1/np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.inputs, units=self.fc1_dims, kernel_initializers = tf.random_uniform(-f1, f1), bias_initializer=tf.random_uniform(-f1, f1))

            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1/np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims, kernel_initializers = tf.random_uniform(-f2, f2), bias_initializer=tf.random_uniform(-f2, f2))
            
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            f3 = 0.003
            mu = tf.layers.dense(layer2_activation, units=self.n_actions, activation='tanh', kernel_initializer=tf.random_uniform(-f3, f3), bias_initializer=tf.random_uniform(-f3, f3))

            self.mu = tf.multiply(mu, self.action_bound)
            
    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.inputs: inputs})  

    def train(self, inputs, gradients):
        self.sess.run(self.optimize, feed_dict={self.inputs:inputs, self.action_gradients:gradients})
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

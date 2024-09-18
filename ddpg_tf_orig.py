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

class Actor(tf.keras.Model):
    def __init__(self, n_actions, action_bound):
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(400, activation='relu')
        self.fc2 = tf.keras.layers.Dense(300, activation='relu')
        self.mu = tf.keras.layers.Dense(n_actions, activation='tanh')
        self.action_bound = action_bound

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        mu = self.mu(x)
        return mu * self.action_bound

class Critic(tf.keras.Model):
    def __init__(self, n_actions):
        super(Critic, self).__init__()
        self.fc1 = tf.keras.layers.Dense(400, activation='relu')
        self.fc2 = tf.keras.layers.Dense(300, activation='relu')
        self.q = tf.keras.layers.Dense(1)

    def call(self, inputs, actions):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = tf.concat([x, actions], axis=-1)
        return self.q(x)

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        self.actor = Actor(n_actions, env.action_space.high)
        self.critic = Critic(n_actions)
        
        self.target_actor = Actor(n_actions, env.action_space.high)
        self.target_critic = Critic(n_actions)

        self.actor_optimizer = tf.keras.optimizers.Adam(alpha)
        self.critic_optimizer = tf.keras.optimizers.Adam(beta)
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())
        else:
            actor_weights = self.actor.get_weights()
            target_actor_weights = self.target_actor.get_weights()
            self.target_actor.set_weights([self.tau * a + (1 - self.tau) * t for a, t in zip(actor_weights, target_actor_weights)])

            critic_weights = self.critic.get_weights()
            target_critic_weights = self.target_critic.get_weights()
            self.target_critic.set_weights([self.tau * c + (1 - self.tau) * t for c, t in zip(critic_weights, target_critic_weights)])

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor(state).numpy()[0]
        mu_prime = mu + self.noise()
        return mu_prime

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        
        critic_value = self.target_critic(new_states, self.target_actor(new_states))
        target = rewards + self.gamma * critic_value * (1 - dones)

        with tf.GradientTape() as tape:
            critic_loss = tf.keras.losses.MSE(target, self.critic(states, actions))
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions_out = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, actions_out))
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        self.update_network_parameters()

    def save_models(self):
        print('Saving models...')
        self.actor.save_weights('actor.weights.h5')  # Updated file extension
        self.critic.save_weights('critic.weights.h5')  # Updated file extension
        self.target_actor.save_weights('target_actor.weights.h5')  # Updated file extension
        self.target_critic.save_weights('target_critic.weights.h5')  # Updated file extension

    def load_models(self):
        self.actor.load_weights('actor.h5')
        self.critic.load_weights('critic.h5')
        self.target_actor.load_weights('target_actor.h5')
        self.target_critic.load_weights('target_critic.h5')

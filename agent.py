import gym
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class Agent(tf.keras.Model):
    __env : gym.Env
    
    def __init__(self):
        super().__init__()
        self.__env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.__dense1 = tf.keras.layers.Dense(10, activation="relu")
        self.__dense2 = tf.keras.layers.Dense(5, activation="tanh")
        self.__dense3 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.__optimizer = tf.keras.optimizers.Nadam(0.01)
    
    def call(self, inputs):
        model = self.__dense1(inputs)
        model = self.__dense2(model)
        model = self.__dense3(model)
        return model

    def play_one_step(self, obs):
        """
        Given an environment, performs a prediction and applies the action to that environment.
        Calculates the gradient vector to determinate which action is better
        """
        with tf.GradientTape() as tape:
            left_probability = self(tf.expand_dims(obs, axis=0))
            action = tf.random.uniform(shape=(1, 1)) > left_probability
            y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_target, left_probability))
            
        grads = tape.gradient(loss, self.trainable_variables)
        #Action = 1 => right
        #Action = 0 => left
        obs, reward, done, truncated, info = self.__env.step(int(action))
        return obs, reward, done, truncated, grads

    def play_multiple_episodes(self, n_episodes, n_max_steps):
        """
        Plays the game several times
        """
        all_rewards =  []
        all_grads = []
        for match in range(n_episodes):
            current_rewards = []
            current_grads = []
            obs, info = self.__env.reset()
            for step in range(n_max_steps):
                obs, reward, done, truncated, grads = self.play_one_step(obs)
                current_rewards.append(reward)
                current_grads.append(grads)
                if done or truncated: break
                
            all_rewards.append(current_rewards)
            all_grads.append(current_grads)
            
        return all_rewards, all_grads
    
    def __discount_rewards(self, rewards, discount_factor):
        """
        Calculate each action's return
        """
        discounted = np.array(rewards)
        for step in range(len(discounted) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_factor
        return discounted
    
    def __discount_and_normalize_rewards(self, all_rewards, discount_factor):
        """
        Discounts and normalizes rewards to know which actions are better
        """
        all_discounted_rewards = []
        for rewards in all_rewards:
            all_discounted_rewards.append(self.__discount_rewards(rewards, discount_factor))
        flatten_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flatten_rewards.mean()
        reward_std = flatten_rewards.std()
        normalized_rewards = []
        for discounted_rewards in all_discounted_rewards:
            normalized_rewards.append((discounted_rewards - reward_mean) / reward_std)
        return normalized_rewards
    
    def train_model(self, n_iterations = 150, n_matches_per_update = 10, n_max_steps = 100, discount_factor = 0.95):
        """
        Trains the model
        :param int n_iterations: Number of epochs
        :param int n_matches_per_update: Number of matches per epoch
        :param int n_max_steps: Max numbers of actions to take
        :param int discount_factor: Discount factor to discount last action's rewards
        """
        for iteration in range(n_iterations):
            print("Iteration", iteration)
            all_rewards, all_grads = self.play_multiple_episodes(n_matches_per_update, n_max_steps)
            all_final_rewards = self.__discount_and_normalize_rewards(all_rewards, discount_factor)
            
            all_mean_grads = []
            for var_index in range(len(self.trainable_variables)):
                grads = []
                for match_index, final_rewards in enumerate(all_final_rewards):
                    for step, final_reward in enumerate(final_rewards):
                        grads.append(final_reward * all_grads[match_index][step][var_index])
                mean_grads = tf.reduce_mean(grads)
                all_mean_grads.append(mean_grads)
            self.__optimizer.apply_gradients(zip(all_mean_grads, self.trainable_variables))

    def play(self):
        """
        Plays the game and visualizes it
        """
        plt.ion()
        obs, info = self.__env.reset()
        fix, ax = plt.subplots()
        img = ax.imshow(self.__env.render())
        it = 1
        while True:
            ax.set_title(f"Iteration: {it}")
            it += 1
            obs, reward, done, truncated, info = self.play_one_step(obs)
            img.set_data(self.__env.render())
            plt.draw()
            plt.pause(0.02)
            if truncated or done: break
        
        plt.ioff()
        plt.show()
    
    def save_model(self, file_path):
        """Save model's weights"""
        self.save_weights(file_path)
    
    def load_model(self, file_path):
        """Load model's weights"""
        dummy = tf.zeros([1, self.__env.observation_space.shape[0]])
        self(dummy)
        self.load_weights(file_path)
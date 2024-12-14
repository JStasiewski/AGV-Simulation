import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from environment import AGVEnvironment

# Build the neural network model
def build_model(input_shape, action_space):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(action_space, activation='linear'))
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse')
    return model

def main():
    env = AGVEnvironment()
    input_dim = len(env.reset())  # observation dimension
    action_space = env.action_space
    
    # Check if the model exists, load it or create a new one
    model_path = "car_dqn.h5"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("No existing model found. Creating a new one.")
        model = build_model(input_dim, action_space)
    
    # Training parameters
    episodes = 500
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    reward_history = []  # To store total rewards per episode
    mean_reward_window = 50  # Window size for mean reward
    
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.sample_action()
            else:
                q_values = model.predict(obs.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0])
            
            # Step the environment
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Compute target
            target = reward
            if not done:
                next_q = model.predict(next_obs.reshape(1, -1), verbose=0)[0]
                target += gamma * np.max(next_q)
            
            # Update Q-values
            current_q = model.predict(obs.reshape(1, -1), verbose=0)
            current_q[0][action] = target
            
            # Train the model
            model.fit(obs.reshape(1, -1), current_q, verbose=0)
            
            obs = next_obs
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # Store the reward
        reward_history.append(total_reward)
        
        # Calculate mean reward over the last `mean_reward_window` episodes
        if len(reward_history) >= mean_reward_window:
            mean_reward = np.mean(reward_history[-mean_reward_window:])
        else:
            mean_reward = np.mean(reward_history)
        
        # Print progress
        print(f"Episode: {ep + 1}, Total Reward: {total_reward}, Mean Reward: {mean_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    # Save the model after training
    model.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()

if __name__ == "__main__":
    main()

from stable_baselines3 import PPO
from env import AGVLineFollowingEnvironment
from stable_baselines3.common.callbacks import BaseCallback
import os
import matplotlib.pyplot as plt

class RewardCallback(BaseCallback):
    """
    Custom callback for plotting average episode reward and its mean during training.
    """
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_timesteps = []
        self.episode_means = []
        self.current_episode_rewards = []

    def _on_step(self) -> bool:
        # Add current reward to the list for the ongoing episode
        self.current_episode_rewards.append(self.locals["rewards"][0])
        
        # Check if an episode is done
        if self.locals["dones"][0]:
            # Calculate mean reward for the episode
            episode_reward = sum(self.current_episode_rewards)
            self.episode_rewards.append(episode_reward)
            self.episode_timesteps.append(self.num_timesteps)
            # Calculate running mean
            running_mean = sum(self.episode_rewards) / len(self.episode_rewards)
            self.episode_means.append(running_mean)
            # Reset for next episode
            self.current_episode_rewards = []
        
        return True

def train_model(existing_model_path=None, total_timesteps=200000):
    """
    Train a new PPO model or continue training an existing one.

    :param existing_model_path: Path to an existing model. If None, a new model will be created.
    :param total_timesteps: Total timesteps to train the model.
    """
    env = AGVLineFollowingEnvironment()
    callback = RewardCallback()

    if existing_model_path and os.path.exists(existing_model_path):
        print(f"Loading existing model from: {existing_model_path}")
        model = PPO.load(existing_model_path, env=env, device="cpu")  # Load model and attach environment
    else:
        print("No existing model found. Creating a new model...")
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")  # Always run on CPU

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the updated or newly trained model
    model.save("ppo_line_follow_model")
    env.close()

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(callback.episode_timesteps, callback.episode_means, label="Mean Reward (Running Average)")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Model trained and saved as 'ppo_line_follow_model'.")

if __name__ == "__main__":
    # Path to the existing model (if available)
    existing_model_path = "ppo_line_follow_model.zip"
    train_model(existing_model_path=existing_model_path, total_timesteps=400000)

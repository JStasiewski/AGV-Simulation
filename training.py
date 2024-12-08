from stable_baselines3 import PPO
from environment import AGVEnvironment

def train_model():
    env = AGVEnvironment()
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=1000000)
    model.save("ppo_agv_model")
    env.close()
    print("Model trained and saved as 'ppo_agv_model'.")

if __name__ == "__main__":
    train_model()

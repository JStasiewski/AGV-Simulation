from stable_baselines3 import PPO
from environment import AGVEnvironment

def simulate_model():
    env = AGVEnvironment()
    model = PPO.load("ppo_agv_model")
    obs = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()

    print("Simulation complete. Goal reached!")
    env.close()

if __name__ == "__main__":
    simulate_model()

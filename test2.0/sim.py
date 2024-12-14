from stable_baselines3 import PPO
from env import AGVLineFollowingEnvironment
import time

def simulate_model():
    env = AGVLineFollowingEnvironment()
    model = PPO.load("ppo_line_follow_model", device="cpu")
    obs = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.02)

    print("Simulation complete. Goal reached or episode ended.")
    env.close()

if __name__ == "__main__":
    simulate_model()
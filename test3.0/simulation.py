import numpy as np
import pyglet
import time
import tensorflow as tf
from environment import AGVEnvironment

def main():
    env = AGVEnvironment()
    model = tf.keras.models.load_model("car_dqn.h5")
    
    obs = env.reset()
    done = False
    
    def update(dt):
        nonlocal obs, done
        if done:
            obs = env.reset()
            done = False
        q_values = model.predict(obs.reshape(1,-1), verbose=0)
        action = np.argmax(q_values[0])
        obs, reward, done, _ = env.step(action)
        env.render()

    pyglet.clock.schedule_interval(update, 1/30.0)
    pyglet.app.run()

if __name__ == "__main__":
    main()

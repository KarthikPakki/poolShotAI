

from pool_env import PoolToolEnv
env = PoolToolEnv()
obs, _ = env.reset()
print(f"Observation: {obs}")
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
env.render()
print(f"Reward: {reward}, Done: {done}")



'''from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from pool_env import PoolToolEnv

env = PoolToolEnv()
check_env(env, warn=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

model.learn(total_timesteps=100000)
model.save("ppo_pooltool")

obs, _ = env.reset()
for _ in range(5):
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break
    print(f"Episode reward: {total_reward}")
    obs, _ = env.reset()

env.close()

'''
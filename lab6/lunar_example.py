import gym
env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42, return_info=True)

# for _ in range(1000):
#    env.render()
#    action = policy(observation)  # User-defined policy function
#    observation, reward, done, info = env.step(action)

#    if done:
#       observation, info = env.reset(return_info=True)
# env.close()
for _ in range(100):
   env.render()
env.close()



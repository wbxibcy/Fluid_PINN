import gym
import numpy as np
import pandas as pd
from gym import spaces
import random
import os
import json

class CO2Env(gym.Env):
    def __init__(self, co2_data):
        super(CO2Env, self).__init__()
        self.co2_data = co2_data
        self.index = 0

        # 根据 CSV 数据计算均值，设置初始阈值
        mean_co2 = np.mean(self.co2_data)
        self.lower_threshold = mean_co2 - 10  # 初始 lower_threshold 为均值 - 10
        self.upper_threshold = mean_co2 + 10  # 初始 upper_threshold 为均值 + 10

        # 动作空间：对 lower 和 upper 做组合调整
        # 0~8: (lower_adj, upper_adj)
        self.action_map = [
            (-5, -5), (-2, 0), (-5, 5),
            (0, -3),  (0, 0),  (0, 3),
            (5, -5), (2, 0), (5, 5)
        ]
        self.action_space = spaces.Discrete(len(self.action_map))
        self.observation_space = spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32)

    def reset(self):
        self.index = 0
        # 使用均值±10设置初始阈值
        mean_co2 = np.mean(self.co2_data)
        self.lower_threshold = mean_co2 - 10
        self.upper_threshold = mean_co2 + 10
        return np.array([self.co2_data[self.index]], dtype=np.float32)

    def step(self, action):
        lower_adj, upper_adj = self.action_map[action]

        # 更新阈值
        self.lower_threshold += lower_adj
        self.upper_threshold += upper_adj
        
        MAX_THRESHOLD_DIFF = 400  # 设置最大差距
        mean_co2 = np.mean(self.co2_data)
        self.lower_threshold = np.clip(self.lower_threshold, mean_co2 - MAX_THRESHOLD_DIFF, mean_co2)
        self.upper_threshold = np.clip(self.upper_threshold, mean_co2, mean_co2 + MAX_THRESHOLD_DIFF)

        # 保证 lower < upper，防止异常
        if self.lower_threshold >= self.upper_threshold:
            self.lower_threshold, self.upper_threshold = self.upper_threshold - 10, self.upper_threshold

        self.index += 1
        done = self.index >= len(self.co2_data)

        if done:
            return np.array([0.0], dtype=np.float32), 0.0, True, {}

        co2_val = self.co2_data[self.index]
        state = np.array([co2_val], dtype=np.float32)

        # 奖励逻辑：是否在阈值范围内
        reward = 1.0 if self.lower_threshold <= co2_val <= self.upper_threshold else -1.0

        return state, reward, done, {}

# 离散化函数
def discretize(value, bins):
    # 根据离散化的边界来给定状态索引
    return np.digitize(value, bins) - 1  # 返回的索引从 0 开始

# 数据读取函数
def load_co2_data(filepath):
    df = pd.read_csv(filepath)
    return df['CO2_Level(ppm)'].values.astype(np.float32)


# 主程序
if __name__ == "__main__":
    # 加载数据
    filepath = os.path.join("../data", "co2_data.csv")
    co2_values = load_co2_data(filepath)

    env = CO2Env(co2_values)

    n_actions = env.action_space.n
    bins = np.linspace(np.min(co2_values), np.max(co2_values), 20)  # 使用数据的最小值和最大值来设置离散化的区间
    n_states = len(bins)

    # 初始化 Q 表：每个状态-动作对的 Q 值
    Q = np.zeros((n_states, n_actions))

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    episodes = 200

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_discrete = discretize(state[0], bins)

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # 随机选择动作
            else:
                action = np.argmax(Q[state_discrete])  # 选择最大 Q 值的动作

            next_state, reward, done, _ = env.step(action)
            next_state_discrete = discretize(next_state[0], bins)

            # Q-Learning 更新公式
            Q[state_discrete][action] += alpha * (
                reward + gamma * np.max(Q[next_state_discrete]) - Q[state_discrete][action]
            )

            state = next_state
            total_reward += reward

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    # 测试
    print("\n✅ 测试策略: ")
    state = env.reset()
    done = False
    while not done:
        state_discrete = discretize(state[0], bins)
        action = np.argmax(Q[state_discrete])  # 获取最佳动作
        state, reward, done, _ = env.step(action)
        print(f"CO2: {state[0]:.2f}, Action: {action}, "
              f"Range: [{env.lower_threshold:.1f}, {env.upper_threshold:.1f}], Reward: {reward}")
    
    config = {
        "lower_threshold": env.lower_threshold,
        "upper_threshold": env.upper_threshold
        }
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

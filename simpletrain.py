import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# 匯入你的遊戲環境
from sawblade_env import SawbladeEnv 

class BasicGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game_env = SawbladeEnv()
        
        # 🌟 核心升級：使用 MultiDiscrete 空間
        # 第一個維度有 4 種選擇 (按鍵)
        # 第二個維度有 10 種選擇 (時間檔位)
        self.action_space = spaces.MultiDiscrete([4, 10])
        
        # 觀察空間保持不變 (12 個浮點數)
        self.observation_space = spaces.Box(low=0, high=2000, shape=(12,), dtype=np.float32)

    def _format_state(self, raw_state):
        obs = np.zeros(12, dtype=np.float32)
        player_pos, sawblades = raw_state
        if player_pos is not None:
            obs[0], obs[1] = player_pos
        for i, (sx, sy) in enumerate(sawblades[:5]):
            obs[2 + i*2] = sx
            obs[3 + i*2] = sy
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._format_state(self.game_env.reset())
        return obs, {} 

    # 🌟 修改 step：解析 AI 丟出來的兩個數字
    def step(self, action):
        # action 現在是一個陣列，例如 [2, 5]
        button_choice = action[0]
        duration_choice = action[1]

        # 1. 決定按鍵 (3 是不動)
        env_action = None if button_choice == 3 else int(button_choice)
        
        # 2. 決定時間：把 0~9 的檔位，轉換成 50ms ~ 500ms
        # 公式： 50 + (檔位 * 50)
        # 例如選 0 -> 50ms (小跳)
        # 例如選 9 -> 500ms (大跳)
        duration_ms = 50 + (int(duration_choice) * 50)

        # 把動作和時間一起傳給遊戲環境
        raw_state, reward, done = self.game_env.step(env_action, duration_ms)
        obs = self._format_state(raw_state)
        
        return obs, float(reward), bool(done), False, {}
# ==========================================
# 🧠 極簡訓練區
# ==========================================
if __name__ == "__main__":
    print("🚀 啟動極簡版 AI 訓練...")
    env = BasicGymEnv()

    # 建立 PPO 模型
    model = PPO("MlpPolicy", env, verbose=1)

    print("🔥 AI 開始遊玩與學習！(按 Ctrl+C 中斷)")
    
    # 讓 AI 先玩 1 萬步測試流程是否順暢
    model.learn(total_timesteps=10000)

    # 存檔
    model.save("simple_ai_model")
    print("🎉 訓練結束並已存檔！")
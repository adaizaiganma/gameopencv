import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# 匯入我們寫好的環境
from sawblade_env import SawbladeEnv


# ==========================================
# 1. 定義標準化的 Gym 環境外殼
# ==========================================
class SawbladeGymEnv(gym.Env):
    def __init__(self):
        super(SawbladeGymEnv, self).__init__()
        self.game_env = SawbladeEnv()

        # 動作空間: 0(左), 1(右), 2(跳), 3(不動)
        self.action_space = spaces.Discrete(4)

        # 觀察空間: [主角X, 主角Y, 鋸1X, 鋸1Y ... 鋸5X, 鋸5Y] 共 12 個浮點數
        self.max_sawblades = 5
        self.observation_space = spaces.Box(
            low=0, high=2000, shape=(2 + self.max_sawblades * 2,), dtype=np.float32
        )

    def _format_state(self, raw_state):
        player_pos, sawblades = raw_state
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        if player_pos is not None:
            obs[0], obs[1] = player_pos

        for i, (sx, sy) in enumerate(sawblades[: self.max_sawblades]):
            obs[2 + i * 2] = sx
            obs[3 + i * 2] = sy
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._format_state(self.game_env.reset())
        return obs, {}

    def step(self, action):
        env_action = None if action == 3 else int(action)
        raw_state, reward, done = self.game_env.step(env_action)
        obs = self._format_state(raw_state)
        return obs, float(reward), bool(done), False, {}


# ==========================================
# 2. 正式開始訓練 AI 大腦
# ==========================================
if __name__ == "__main__":
    # 建立資料夾用來存放 Log 和模型存檔
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("檢查環境標準格式是否正確...")
    # 用 Monitor 包裝環境，這樣 SB3 才會幫我們記錄每一局的得分與存活時間
    env = Monitor(SawbladeGymEnv(), log_dir)
    check_env(env)
    print("✅ 環境檢查通過！")

    # 設定自動存檔 (每走 1000 步就自動存一個檔，避免當機心血全毀)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path="./models/", name_prefix="sawblade_model"
    )

    # 建立 PPO 模型，並開啟 tensorboard 支援
    print("\n🤖 正在初始化 PPO 強化學習模型...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    print("\n🔥 開始無盡的訓練輪迴！(按 Ctrl+C 可以強制停止)")
    try:
        # 開始訓練 10 萬步
        model.learn(total_timesteps=100000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\n🛑 訓練已被手動中斷。")

    # 儲存最終的 AI 大腦
    print("💾 正在儲存最終模型到 'sawblade_final_model.zip'...")
    model.save("sawblade_final_model")

    print("🎉 訓練結束！你可以隨時載入這個模型來讓 AI 玩給你看！")

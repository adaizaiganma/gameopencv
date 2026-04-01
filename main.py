import subprocess
import cv2
import numpy as np
import math
import time
import glob
import os

# ==========================================
# 🌟 1. 設定數值
# ==========================================
DEVICE_ADDRESS = "127.0.0.1:62001"
PLAYER_IMG_DIR = "player_img"

# 🌟 新增：UI 圖片的路徑
UI_MENU_IMG = "ui_img/menu_start.png"
UI_GAMEOVER_IMG = "ui_img/gameover_retry.png"

# 按鈕座標
BTN_LEFT = (150, 800)
BTN_RIGHT = (350, 800)
BTN_JUMP = (600, 800)

LOWER_COLOR = np.array([0, 0, 0])
UPPER_COLOR = np.array([255, 255, 255])
GROUND_Y = 600
DEATH_RADIUS = 45
# ==========================================


# (GameLogicEngine 保持不變，這裡省略以節省版面)
class GameLogicEngine:
    def __init__(self):
        self.ground_y = GROUND_Y
        self.death_radius = DEATH_RADIUS
        self.dodging_sawblades = []
        self.score = 0

    def check_state(self, player_pos, sawblade_positions):
        if not player_pos:
            return False, 0
        px, py = player_pos
        reward = 0
        is_dead = False
        for sx, sy in sawblade_positions:
            if math.hypot(px - sx, py - sy) < self.death_radius:
                return True, -100
        is_in_air = py < (self.ground_y - 20)
        if is_in_air:
            for sx, sy in sawblade_positions:
                if sy > py and abs(px - sx) < 30 and sx not in self.dodging_sawblades:
                    self.dodging_sawblades.append(sx)
        else:
            if len(self.dodging_sawblades) > 0:
                reward += len(self.dodging_sawblades) * 10
                self.score += len(self.dodging_sawblades)
                self.dodging_sawblades.clear()
        reward += 0.1
        return is_dead, reward


class SawbladeEnv:
    def __init__(self):
        self.logic = GameLogicEngine()
        self.player_templates = []

        # 載入主角圖片
        template_files = glob.glob(os.path.join(PLAYER_IMG_DIR, "*.png"))
        for f in template_files:
            temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if temp is not None:
                self.player_templates.append(
                    (temp, temp.shape[1], temp.shape[0], os.path.basename(f))
                )

        # 🌟 新增：載入 UI 模板
        self.ui_menu = cv2.imread(UI_MENU_IMG, cv2.IMREAD_GRAYSCALE)
        self.ui_gameover = cv2.imread(UI_GAMEOVER_IMG, cv2.IMREAD_GRAYSCALE)
        if self.ui_menu is None or self.ui_gameover is None:
            print("⚠️ 警告：找不到 UI 圖片，請確認 ui_img 資料夾內的檔案名稱與路徑。")

    def _take_action(self, action):
        duration = 50 if action == 2 else 250
        coords = {0: BTN_LEFT, 1: BTN_RIGHT, 2: BTN_JUMP}.get(action, BTN_LEFT)
        x, y = coords
        command = (
            f"adb -s {DEVICE_ADDRESS} shell input swipe {x} {y} {x} {y} {duration}"
        )
        subprocess.run(command, shell=True)

    # 🌟 新增：自由點擊畫面上任何座標的功能
    def _tap_screen(self, x, y):
        command = f"adb -s {DEVICE_ADDRESS} shell input tap {x} {y}"
        subprocess.run(command, shell=True)

    def _get_screen(self):
        command = f"adb -s {DEVICE_ADDRESS} exec-out screencap -p"
        try:
            pipe = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            image_bytes = pipe.stdout.read()
            if not image_bytes:
                return None
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except:
            return None

    # 🌟 新增：檢查目前是哪個畫面，並回傳需要點擊的座標
    def _check_ui_state(self, screen):
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # 檢查是否在死亡結算畫面
        if self.ui_gameover is not None:
            res = cv2.matchTemplate(gray_screen, self.ui_gameover, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.7:  # 信心門檻
                h, w = self.ui_gameover.shape
                return "GAMEOVER", (max_loc[0] + w // 2, max_loc[1] + h // 2)

        # 檢查是否在主選單
        if self.ui_menu is not None:
            res = cv2.matchTemplate(gray_screen, self.ui_menu, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.7:
                h, w = self.ui_menu.shape
                return "MENU", (max_loc[0] + w // 2, max_loc[1] + h // 2)

        return "PLAYING", None

    def _extract_features(self, screen):
        """從畫面中找出主角與電鋸的座標"""
        player_pos = None
        sawblade_positions = []

        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # 🌟 多模板匹配邏輯
        best_player_val = -1.0
        best_player_pos = None

        for temp_img, temp_w, temp_h, temp_name in self.player_templates:
            res = cv2.matchTemplate(gray_screen, temp_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # 如果這個姿態的分數更高，就記錄下來
            if max_val > best_player_val:
                best_player_val = max_val
                # 計算中心點
                best_player_pos = (max_loc[0] + temp_w // 2, max_loc[1] + temp_h // 2)

        # 設定信心門檻 (只要最高分的那個姿態大於 0.6 就當作找到了)
        if best_player_val > 0.6:
            player_pos = best_player_pos

        # 找電鋸 (HSV 過濾)
        h, w = screen.shape[:2]
        roi_bottom = int(h * 0.75)
        img_detect = screen.copy()
        cv2.rectangle(img_detect, (0, roi_bottom), (w, h), (0, 0, 0), -1)

        hsv = cv2.cvtColor(img_detect, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 0 and (area / (np.pi * (radius**2))) > 0.6:
                    sawblade_positions.append((int(x), int(y)))

        return player_pos, sawblade_positions

    # 🌟 核心升級：會自動點擊按鈕直到進入遊戲為止的 reset
    def reset(self):
        print("🔄 環境準備重置，正在偵測畫面狀態...")

        # 進入一個等待迴圈，直到確認遊戲正式開始為止
        while True:
            screen = self._get_screen()
            if screen is None:
                continue

            ui_state, click_coords = self._check_ui_state(screen)

            if ui_state == "GAMEOVER":
                print(f"💀 偵測到死亡結算畫面，點擊重試按鈕 {click_coords}...")
                self._tap_screen(click_coords[0], click_coords[1])
                time.sleep(1.5)  # 等待過場動畫

            elif ui_state == "MENU":
                print(f"🏠 偵測到主選單，點擊開始遊戲 {click_coords}...")
                self._tap_screen(click_coords[0], click_coords[1])
                time.sleep(1.5)

            elif ui_state == "PLAYING":
                # 畫面中沒有 UI 按鈕，代表已經進入遊戲畫面！
                print("🎮 成功進入遊戲畫面，初始化邏輯！")
                break

        self.logic = GameLogicEngine()
        screen = self._get_screen()
        state = self._extract_features(screen) if screen is not None else (None, [])
        return state

    def step(self, action):
        if action is not None:
            self._take_action(action)
            time.sleep(0.1)

        screen = self._get_screen()
        if screen is None:
            return (None, []), 0, False

        player_pos, sawblades = self._extract_features(screen)
        state = (player_pos, sawblades)
        is_dead, reward = self.logic.check_state(player_pos, sawblades)
        return state, reward, is_dead


# 測試區
if __name__ == "__main__":
    import random

    env = SawbladeEnv()

    # 測試自動重置功能
    state = env.reset()

    print("\n隨機亂跳測試開始...")
    for step_num in range(30):
        action = random.choice([0, 1, 2, None])
        next_state, reward, done = env.step(action)
        print(f"Step {step_num}: 執行 {action}, 獎勵 {reward:.1f}, 死亡? {done}")

        if done:
            print(">>> 觸發死亡，呼叫 reset() 測試自動恢復功能 <<<")
            state = env.reset()  # 死亡後再次呼叫 reset，它應該要自動幫你按 Retry！

import subprocess
import cv2
import numpy as np
import math
import time
import glob
import os

# ==========================================
# 🌟 1. 設定區域 (請填入你測試成功的數值)
# ==========================================
DEVICE_ADDRESS = "127.0.0.1:16384"
PLAYER_IMG_DIR = "player_img"
UI_MENU_IMG = "ui_img/menu_start.png"
UI_GAMEOVER_IMG = "ui_img/gameover_retry.png"

# 按鈕座標
BTN_LEFT = (170, 1700)
BTN_RIGHT = (400, 1700)
BTN_JUMP = (800, 1700)

LOWER_COLOR = np.array([0, 38, 59])
UPPER_COLOR = np.array([179, 113, 135])
GROUND_Y = 600
DEATH_RADIUS = 45
# ==========================================


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

        # 1. 死亡判定
        for sx, sy in sawblade_positions:
            if math.hypot(px - sx, py - sy) < self.death_radius:
                return True, -100

        # 2. 閃避判定
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

        # 3. 存活獎勵
        reward += 0.1
        return is_dead, reward


class SawbladeEnv:
    def __init__(self):
        self.logic = GameLogicEngine()
        self.player_templates = []

        # 載入所有主角圖片
        for f in glob.glob(os.path.join(PLAYER_IMG_DIR, "*.png")):
            temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if temp is not None:
                self.player_templates.append((temp, temp.shape[1], temp.shape[0]))

        # 載入 UI 圖片
        self.ui_menu = cv2.imread(UI_MENU_IMG, cv2.IMREAD_GRAYSCALE)
        self.ui_gameover = cv2.imread(UI_GAMEOVER_IMG, cv2.IMREAD_GRAYSCALE)

    def _take_action(self, action):
        duration = 50 if action == 2 else 250
        coords = {0: BTN_LEFT, 1: BTN_RIGHT, 2: BTN_JUMP}.get(action, BTN_LEFT)
        x, y = coords
        subprocess.run(
            f"adb -s {DEVICE_ADDRESS} shell input swipe {x} {y} {x} {y} {duration}",
            shell=True,
        )

    def _get_screen(self):
        try:
            pipe = subprocess.Popen(
                f"adb -s {DEVICE_ADDRESS} exec-out screencap -p",
                stdout=subprocess.PIPE,
                shell=True,
            )
            bytes_data = pipe.stdout.read()
            if not bytes_data:
                return None
            return cv2.imdecode(
                np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        except:
            return None

    def _check_ui_state(self, screen):
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        if self.ui_gameover is not None:
            _, max_val, _, _ = cv2.minMaxLoc(
                cv2.matchTemplate(gray, self.ui_gameover, cv2.TM_CCOEFF_NORMED)
            )
            if max_val > 0.7:
                return "GAMEOVER"
        if self.ui_menu is not None:
            _, max_val, _, _ = cv2.minMaxLoc(
                cv2.matchTemplate(gray, self.ui_menu, cv2.TM_CCOEFF_NORMED)
            )
            if max_val > 0.7:
                return "MENU"
        return "PLAYING"

    def _extract_features(self, screen):
        player_pos = None
        sawblade_positions = []
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # 找主角 (多模板)
        best_val = -1.0
        for temp_img, w, h in self.player_templates:
            _, max_val, _, max_loc = cv2.minMaxLoc(
                cv2.matchTemplate(gray, temp_img, cv2.TM_CCOEFF_NORMED)
            )
            if max_val > best_val:
                best_val = max_val
                best_pos = (max_loc[0] + w // 2, max_loc[1] + h // 2)
        if best_val > 0.6:
            player_pos = best_pos

        # 找電鋸 (HSV)
        h, w = screen.shape[:2]
        img_detect = screen.copy()
        cv2.rectangle(img_detect, (0, int(h * 0.75)), (w, h), (0, 0, 0), -1)
        mask = cv2.inRange(
            cv2.cvtColor(img_detect, cv2.COLOR_BGR2HSV), LOWER_COLOR, UPPER_COLOR
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 0 and (area / (np.pi * (radius**2))) > 0.6:
                    sawblade_positions.append((int(x), int(y)))
        return player_pos, sawblade_positions

    def reset(self):
        print("🔄 環境重置：等待並導航回遊戲中...")
        while True:
            screen = self._get_screen()
            if screen is None:
                continue
            ui_state = self._check_ui_state(screen)
            if ui_state in ["GAMEOVER", "MENU"]:
                self._take_action(2)  # 按下跳躍鍵確認
                time.sleep(1.5)
            elif ui_state == "PLAYING":
                break

        self.logic = GameLogicEngine()
        screen = self._get_screen()
        return self._extract_features(screen) if screen is not None else (None, [])

    def step(self, action):
        if action is not None:
            self._take_action(action)
            time.sleep(0.1)
        screen = self._get_screen()
        if screen is None:
            return (None, []), 0, False
        player_pos, sawblades = self._extract_features(screen)
        is_dead, reward = self.logic.check_state(player_pos, sawblades)
        return (player_pos, sawblades), reward, is_dead

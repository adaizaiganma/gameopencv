import scrcpy
import cv2
import numpy as np
import math
import time
import glob
import os
import threading

# ==========================================
# 🌟 1. 設定區域 (請填入你測試成功的數值)
# ==========================================
DEVICE_ADDRESS = "127.0.0.1:16384"
PLAYER_IMG_DIR = "player_img"
UI_MENU_IMG = "ui_img/menu_start.png"
UI_GAMEOVER_IMG = "ui_img/gameover_retry.png"

# 按鈕座標

BTN_LEFT = (85, 850)
BTN_RIGHT = (185, 1400)
BTN_JUMP = (400, 1400)

GROUND_Y = 600
DEATH_RADIUS = 45

# 🔴 危險電鋸 (原色，跳過會給分)
LOWER_COLOR_1 = np.array([0, 76, 104])       
UPPER_COLOR_1 = np.array([7, 165, 255]) 

LOWER_COLOR_2 = np.array([41, 9, 134])
UPPER_COLOR_2 = np.array([93, 255, 255])


# ==========================================

class GameLogicEngine:
    def __init__(self):
        self.ground_y = GROUND_Y
        self.death_radius = DEATH_RADIUS
        self.dodging_sawblades = [] 
        self.score = 0

    def check_state(self, player_pos, unscored_sawblades, scored_sawblades):
        if not player_pos: return False, 0
        px, py = player_pos
        reward = 0
        is_dead = False

        all_deadly_sawblades = unscored_sawblades + scored_sawblades

        # 1. 死亡判定
        for sx, sy in all_deadly_sawblades:
            if math.hypot(px - sx, py - sy) < self.death_radius:
                return True, -100 

        # 2. 閃避判定
        is_in_air = py < (self.ground_y - 20) 
        if is_in_air:
            for sx, sy in unscored_sawblades:
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
        # 載入圖片模板
        self.player_templates = []
        for f in glob.glob(os.path.join(PLAYER_IMG_DIR, "*.png")):
            temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if temp is not None:
                self.player_templates.append((temp, temp.shape[1], temp.shape[0]))
                
        self.ui_menu = cv2.imread(UI_MENU_IMG, cv2.IMREAD_GRAYSCALE)
        self.ui_gameover = cv2.imread(UI_GAMEOVER_IMG, cv2.IMREAD_GRAYSCALE)

        # 🌟 初始化 Scrcpy 串流引擎
        print("🚀 啟動 Scrcpy 光速視覺引擎與控制通道...")
        self.latest_frame = None
        self.client = scrcpy.Client(device=DEVICE_ADDRESS)
        self.client.add_listener(scrcpy.EVENT_FRAME, self._on_frame)
        
        # 在背景啟動串流
        self.client.start(threaded=True)
        
        # 等待第一張畫面進來，確保 AI 不會瞎著眼開局
        print("⏳ 等待畫面串流同步...")
        while self.latest_frame is None:
            time.sleep(0.1)
        print("✅ 畫面同步完成！")

    def _on_frame(self, frame):
        """背景執行緒：不斷把最新的畫面更新到變數中"""
        if frame is not None:
            self.latest_frame = frame

    def _get_screen(self):
        """AI 拿畫面時，直接回傳記憶體裡最新的一張，耗時 0 毫秒！"""
        return self.latest_frame.copy() if self.latest_frame is not None else None

    def _take_action(self, action, duration_ms):
        """⚡ 光速點擊：直接透過 Scrcpy 注入觸控事件 (0 延遲)"""
        coords = {0: BTN_LEFT, 1: BTN_RIGHT, 2: BTN_JUMP}.get(action, BTN_LEFT)
        x, y = coords
        
        # 模擬手指按下
        self.client.control.touch(x, y, scrcpy.ACTION_DOWN)
        # 等待按壓時間 (支援 AI 決定的長短按)
        time.sleep(duration_ms / 1000.0)
        # 模擬手指鬆開
        self.client.control.touch(x, y, scrcpy.ACTION_UP)

    def _check_ui_state(self, screen):
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        if self.ui_menu is not None:
            _, max_val, _, _ = cv2.minMaxLoc(cv2.matchTemplate(gray, self.ui_menu, cv2.TM_CCOEFF_NORMED))
            if max_val > 0.7: return "MENU"
        if self.ui_gameover is not None:
            _, max_val, _, _ = cv2.minMaxLoc(cv2.matchTemplate(gray, self.ui_gameover, cv2.TM_CCOEFF_NORMED))
            if max_val > 0.7: return "GAMEOVER"
        return "PLAYING"

    def _extract_features(self, screen):
        player_pos = None
        unscored_sawblades = [] 
        scored_sawblades = []   
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        best_val = -1.0
        for temp_img, w, h in self.player_templates:
            _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(gray, temp_img, cv2.TM_CCOEFF_NORMED))
            if max_val > best_val:
                best_val = max_val
                best_pos = (max_loc[0] + w//2, max_loc[1] + h//2)
        if best_val > 0.6: player_pos = best_pos

        h, w = screen.shape[:2]
        img_detect = screen.copy()
        cv2.rectangle(img_detect, (0, int(h * 0.75)), (w, h), (0, 0, 0), -1)
        hsv = cv2.cvtColor(img_detect, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, LOWER_COLOR_1, UPPER_COLOR_1)
        contours_1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_1:
            if cv2.contourArea(cnt) > 50:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 0 and (cv2.contourArea(cnt) / (np.pi * (radius ** 2))) > 0.6: 
                    unscored_sawblades.append((int(x), int(y)))

        mask2 = cv2.inRange(hsv, LOWER_COLOR_2, UPPER_COLOR_2)
        contours_2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_2:
            if cv2.contourArea(cnt) > 50:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 0 and (cv2.contourArea(cnt) / (np.pi * (radius ** 2))) > 0.6: 
                    scored_sawblades.append((int(x), int(y)))

        return player_pos, unscored_sawblades, scored_sawblades

    def reset(self):
        print("🔄 環境重置：執行 UI 狀態機檢查...")
        while True:
            screen = self._get_screen()
            if screen is None: continue
            
            ui_state = self._check_ui_state(screen)
            
            if ui_state == "MENU":
                print("🏠 主選單：光速點擊跳躍鍵...")
                self._take_action(2, 50) 
                time.sleep(1.5) 
                
            elif ui_state == "GAMEOVER":
                print("💀 死亡畫面：光速點擊重新開始...")
                self._take_action(2, 50)
                time.sleep(1.5)
                
            elif ui_state == "PLAYING":
                break
                
        self.logic = GameLogicEngine()
        screen = self._get_screen()
        if screen is not None:
            player_pos, unscored, scored = self._extract_features(screen)
            return player_pos, (unscored + scored)
        return None, []

    def step(self, action, duration_ms=250):
        if action is not None:
            self._take_action(action, duration_ms)
            # 因為按鍵現在是 0 延遲，我們稍微等一下讓遊戲畫面有時間更新
            time.sleep(0.02) 
            
        screen = self._get_screen()
        if screen is None: return (None, []), 0, False
        
        player_pos, unscored, scored = self._extract_features(screen)
        is_dead, reward = self.logic.check_state(player_pos, unscored, scored)
        all_sawblades_for_ai_eyes = unscored + scored
        
        return (player_pos, all_sawblades_for_ai_eyes), reward, is_dead
        
    def close(self):
        """關閉環境時，安全地關閉串流"""
        if hasattr(self, 'client'):
            self.client.stop()
import cv2
import numpy as np
import subprocess
import time
import os

# ==========================================
# 🌟 1. 設定區域
# ==========================================
DEVICE_ADDRESS = "127.0.0.1:16384" 
UI_MENU_IMG = "ui_img/menu_start.png"
UI_GAMEOVER_IMG = "ui_img/gameover_retry.png"
THRESHOLD = 0.7  # 信心門檻

def _get_screen():
        """🚀 高速版：支援任意版本 Android 檔頭的截圖"""
        try:
            command = f"adb -s {DEVICE_ADDRESS} exec-out screencap"
            pipe = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            image_bytes = pipe.stdout.read()
            
            if not image_bytes: 
                return None

            # 讀取寬度和高度
            width = int.from_bytes(image_bytes[0:4], byteorder='little')
            height = int.from_bytes(image_bytes[4:8], byteorder='little')
            
            # 🌟 核心修正：自動反推檔頭大小 (Header Size)
            expected_image_size = width * height * 4
            header_size = len(image_bytes) - expected_image_size
            
            # 從正確的位置開始讀取像素數據
            img_data = image_bytes[header_size:]
            
            # 轉換為 numpy 陣列並重塑形狀
            img_np = np.frombuffer(img_data, dtype=np.uint8)
            img_rgba = img_np.reshape((height, width, 4))
            
            # 轉成 BGR 給 OpenCV 使用
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            
            return img_bgr
            
        except Exception as e:
            print(f"高速截圖發生錯誤: {e}")
            return None
        
def test_ui_radar():
    print("🚀 啟動 UI 狀態監控雷達...")
    
    # 檢查模板檔案是否存在
    if not os.path.exists(UI_MENU_IMG) or not os.path.exists(UI_GAMEOVER_IMG):
        print("❌ 錯誤：找不到 UI 模板圖片！請檢查路徑。")
        return

    # 載入模板
    ui_menu = cv2.imread(UI_MENU_IMG, cv2.IMREAD_GRAYSCALE)
    ui_gameover = cv2.imread(UI_GAMEOVER_IMG, cv2.IMREAD_GRAYSCALE)

    print("👉 請在彈出的視窗中觀察分數。按『q』鍵退出。")

    while True:
        start_time = time.time()
        
        # 1. 抓取畫面
        screen = _get_screen()
        if screen is None:
            continue
            
        display_img = screen.copy()
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        current_state = "PLAYING"
        menu_score = 0
        gameover_score = 0
        
        # 2. 檢測 Menu
        res_menu = cv2.matchTemplate(gray, ui_menu, cv2.TM_CCOEFF_NORMED)
        _, menu_score, _, menu_loc = cv2.minMaxLoc(res_menu)
        
        if menu_score > THRESHOLD:
            current_state = "MENU"
            # 畫出偵測到的框框
            h, w = ui_menu.shape
            cv2.rectangle(display_img, menu_loc, (menu_loc[0]+w, menu_loc[1]+h), (0, 255, 0), 3)

        # 3. 檢測 Game Over (只有在不是 Menu 時才測，避免衝突)
        if current_state != "MENU":
            res_gameover = cv2.matchTemplate(gray, ui_gameover, cv2.TM_CCOEFF_NORMED)
            _, gameover_score, _, gameover_loc = cv2.minMaxLoc(res_gameover)
            
            if gameover_score > THRESHOLD:
                current_state = "GAMEOVER"
                h, w = ui_gameover.shape
                cv2.rectangle(display_img, gameover_loc, (gameover_loc[0]+w, gameover_loc[1]+h), (0, 0, 255), 3)

        # 4. 在畫面上繪製即時數據面板
        panel_color = (0, 0, 0)
        text_color = (255, 255, 255)
        if current_state == "MENU": panel_color = (0, 150, 0)
        elif current_state == "GAMEOVER": panel_color = (0, 0, 150)
        
        # 畫一個半透明的背景底框讓字比較清楚
        cv2.rectangle(display_img, (10, 10), (400, 150), panel_color, -1)
        
        cv2.putText(display_img, f"STATE: {current_state}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
        
        # 🌟 關鍵 Debug 資訊：顯示實際算出來的分數
        cv2.putText(display_img, f"Menu Score: {menu_score:.3f} / {THRESHOLD}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if menu_score > THRESHOLD else (200, 200, 200), 2)
                    
        cv2.putText(display_img, f"GameOver Score: {gameover_score:.3f} / {THRESHOLD}", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if gameover_score > THRESHOLD else (200, 200, 200), 2)

        # 計算並顯示 FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(display_img, f"FPS: {fps:.1f}", (10, display_img.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 顯示畫面 (縮小一點避免超出螢幕)
        display_img_resized = cv2.resize(display_img, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow("UI State Radar", display_img_resized)

        # 按 Q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_ui_radar()
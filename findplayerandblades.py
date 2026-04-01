import cv2
import numpy as np
import subprocess
import time
import glob
import os

# ==========================================
# 🌟 1. 設定區域 (請確保這裡的數值與你 env 裡的一致)
# ==========================================
DEVICE_ADDRESS = "127.0.0.1:16384" 
PLAYER_IMG_DIR = "player_img"  

# 🔴 危險電鋸 (原色)
LOWER_COLOR_1 = np.array([0, 76, 104])       
UPPER_COLOR_1 = np.array([7, 165, 255]) 

# 🟢 安全電鋸 (變色後，請換成你實際測出的數值)
LOWER_COLOR_2 = np.array([0, 0, 0])      
UPPER_COLOR_2 = np.array([255, 255, 255])

# ==========================================

def get_screen():
    """使用最穩定的自動算檔頭 ADB 高速截圖"""
    try:
        command = f"adb -s {DEVICE_ADDRESS} exec-out screencap"
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        image_bytes = pipe.stdout.read()
        
        if not image_bytes: return None

        width = int.from_bytes(image_bytes[0:4], byteorder='little')
        height = int.from_bytes(image_bytes[4:8], byteorder='little')
        
        expected_size = width * height * 4
        header_size = len(image_bytes) - expected_size
        img_data = image_bytes[header_size:]
        
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        img_rgba = img_np.reshape((height, width, 4))
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        return img_bgr
    except:
        return None

def test_vision():
    print("🚀 啟動綜合視覺雷達...")

    # 1. 載入主角模板
    player_templates = []
    for f in glob.glob(os.path.join(PLAYER_IMG_DIR, "*.png")):
        temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if temp is not None:
            player_templates.append((temp, temp.shape[1], temp.shape[0]))
    
    if not player_templates:
        print(f"⚠️ 警告：在 '{PLAYER_IMG_DIR}' 找不到主角圖片，主角追蹤將失效！")

    print("👉 請在彈出的視窗中觀察標記狀況。按『q』鍵退出。")

    while True:
        start_time = time.time()
        screen = get_screen()
        if screen is None: continue

        display_img = screen.copy()
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        h, w = screen.shape[:2]

        # ==========================================
        # 🕵️‍♂️ 偵測主角 (多模板匹配)
        # ==========================================
        best_val = -1.0
        best_pos = None
        best_w, best_h = 0, 0
        
        for temp_img, tw, th in player_templates:
            res = cv2.matchTemplate(gray, temp_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_pos = (max_loc[0], max_loc[1]) # 左上角
                best_w, best_h = tw, th

        # 如果分數大於 0.6，畫出「藍色」準心與框框
        if best_val > 0.6 and best_pos:
            center_x = best_pos[0] + best_w // 2
            center_y = best_pos[1] + best_h // 2
            
            # 畫框框
            cv2.rectangle(display_img, best_pos, (best_pos[0]+best_w, best_pos[1]+best_h), (255, 255, 0), 2)
            # 畫中心點
            cv2.circle(display_img, (center_x, center_y), 4, (255, 255, 0), -1)
            # 顯示分數
            cv2.putText(display_img, f"Player: {best_val:.2f}", (best_pos[0], best_pos[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ==========================================
        # 🪚 偵測電鋸 (雙色 HSV)
        # ==========================================
        # 設定 ROI (避開底部 UI)
        img_detect = screen.copy()
        roi_bottom = int(h * 0.75)
        cv2.rectangle(img_detect, (0, roi_bottom), (w, h), (0, 0, 0), -1)
        
        hsv = cv2.cvtColor(img_detect, cv2.COLOR_BGR2HSV)
        
        # 找危險電鋸 (原色)
        mask1 = cv2.inRange(hsv, LOWER_COLOR_1, UPPER_COLOR_1)
        contours_1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        danger_count = 0
        for cnt in contours_1:
            if cv2.contourArea(cnt) > 50:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 0 and (cv2.contourArea(cnt) / (np.pi * (radius ** 2))) > 0.6: 
                    danger_count += 1
                    cv2.circle(display_img, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                    cv2.putText(display_img, "DANGER", (int(x)-25, int(y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 找安全電鋸 (變色)
        mask2 = cv2.inRange(hsv, LOWER_COLOR_2, UPPER_COLOR_2)
        contours_2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        safe_count = 0
        for cnt in contours_2:
            if cv2.contourArea(cnt) > 50:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 0 and (cv2.contourArea(cnt) / (np.pi * (radius ** 2))) > 0.6: 
                    safe_count += 1
                    cv2.circle(display_img, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    cv2.putText(display_img, "SAFE", (int(x)-15, int(y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ==========================================
        # 📊 繪製資訊面板
        # ==========================================
        fps = 1.0 / (time.time() - start_time)
        
        cv2.rectangle(display_img, (10, 10), (350, 110), (0, 0, 0), -1)
        cv2.putText(display_img, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Danger Saws: {danger_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_img, f"Safe Saws: {safe_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 畫一條黃線表示 ROI 界線
        cv2.line(display_img, (0, roi_bottom), (w, roi_bottom), (0, 255, 255), 2)

        # 顯示縮小版畫面
        display_img_resized = cv2.resize(display_img, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow("Omni-Vision Radar", display_img_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_vision()
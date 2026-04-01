import cv2
import numpy as np

# ==========================================
# 🌟 1. 配置測試圖片路徑
# ==========================================
# 請替換為你實際擁有的圖片路徑
FRAME_PATH = "frames/frame_0624.png"        # 你的遊戲畫面大圖 (例如某個影格)
TEMPLATE_PATH = "ui_img/gameover_retry.png" # 你的主角截圖小圖

def test_single_template_matching():
    print("🔍 開始單模板匹配測試...")

    # 1. 讀取圖片 
    # 大圖我們讀取彩色的用來顯示，但計算時會轉灰階
    frame = cv2.imread(FRAME_PATH)
    # 小圖(模板)直接以灰階模式讀取，減少顏色干擾，提高穩定度
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

    # 防呆機制：檢查圖片是否有成功讀取
    if frame is None:
        print(f"❌ 錯誤: 找不到遊戲畫面 '{FRAME_PATH}'")
        return
    if template is None:
        print(f"❌ 錯誤: 找不到模板圖片 '{TEMPLATE_PATH}'")
        return

    # 將遊戲畫面大圖也轉換為灰階
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 獲取模板的寬度和高度 (之後畫綠色框框會用到)
    h, w = template.shape

    # ==========================================
    # 🌟 2. 執行核心演算法
    # ==========================================
    # 使用 TM_CCOEFF_NORMED 算法，回傳值在 -1 到 1 之間，越接近 1 越相似
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

    # minMaxLoc 會在一大堆結果中，幫我們找出「最低分」與「最高分」的數值與座標
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"📊 匹配相似度最高分: {max_val:.4f}")

    # ==========================================
    # 🌟 3. 繪製結果與判定
    # ==========================================
    # 設定信心門檻 (大於這個分數我們才承認有找到)
    threshold = 0.9 
    
    # 複製一張原圖用來畫畫，以免破壞原始圖片數據
    display_img = frame.copy()

    if max_val >= threshold:
        print(f"✅ 成功找到目標！位於座標: {max_loc} (X, Y)")
        
        # 計算框框的左上角與右下角
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 畫一個綠色、粗細為 2 的矩形框
        cv2.rectangle(display_img, top_left, bottom_right, (0, 255, 0), 2)

        # 在框框上方標示分數
        text = f"Match: {max_val:.2f}"
        cv2.putText(display_img, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print("⚠️ 未能找到目標 (分數低於設定的門檻)。")
        
        # 在畫面上印出大大的紅字警告
        cv2.putText(display_img, f"LOST (Max: {max_val:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ==========================================
    # 🌟 4. 顯示圖片
    # ==========================================
    print("👉 請在彈出的 OpenCV 視窗中按『任意鍵』關閉測試。")
    cv2.imshow("Template Matching Test", display_img)
    
    # waitKey(0) 代表無限期等待，直到你在視窗上按下鍵盤任意鍵
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_single_template_matching()
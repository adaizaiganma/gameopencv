import cv2
import numpy as np
import glob
import os

FRAME_DIR = "frames"

# 🌟 定義第一種顏色的範圍 (你已經找出來的)
lower_color_1 = np.array([0, 76, 104])
upper_color_1 = np.array([7, 165, 255])

lower_color_2 = np.array([41, 9, 134])
upper_color_2 = np.array([93, 255, 255])
def detect_sawblades():
    files = sorted(glob.glob(os.path.join(FRAME_DIR, "*.png")))
    if not files:
        print("錯誤：資料夾內沒有圖片")
        return

    print("開始雙色電鋸追蹤測試...")

    for f in files:
        img = cv2.imread(f)
        h, w = img.shape[:2]

        roi_bottom = int(h * 0.7)
        img_draw = img.copy()
        img_detect = img.copy()
        cv2.rectangle(img_detect, (0, roi_bottom), (w, h), (0, 0, 0), -1)

        hsv = cv2.cvtColor(img_detect, cv2.COLOR_BGR2HSV)

        # ==========================================
        # 🌟 核心修改：雙色遮罩合併
        # ==========================================
        # 1. 產生第一種顏色的遮罩
        mask1 = cv2.inRange(hsv, lower_color_1, upper_color_1)
        
        # 2. 產生第二種顏色的遮罩
        mask2 = cv2.inRange(hsv, lower_color_2, upper_color_2)
        
        # 3. 將兩個遮罩聯集 (OR)。只要在 mask1 或 mask2 中是白色的，就會保留
        mask = cv2.bitwise_or(mask1, mask2)
        # ==========================================

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sawblade_positions = []

        for cnt in contours:
            contour_area = cv2.contourArea(cnt)
            if contour_area > 50:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius == 0: continue
                circle_area = np.pi * (radius**2)
                confidence = contour_area / circle_area

                if confidence > 0.65:
                    center = (int(x), int(y))
                    radius = int(radius)
                    sawblade_positions.append(center)
                    cv2.circle(img_draw, center, radius, (0, 0, 255), 2)
                    
                    text = f"({center[0]},{center[1]}) C:{confidence:.2f}"
                    cv2.putText(img_draw, text, (center[0] + 10, center[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(img_draw, f"Sawblades: {len(sawblade_positions)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.line(img_draw, (0, roi_bottom), (w, roi_bottom), (0, 255, 255), 2)

        # 顯示處理後的對照圖 (讓你確認兩種顏色的電鋸是否都成功變成白塊)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_img = np.hstack((img_draw, mask_bgr))
        
        display_img = cv2.resize(combined_img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Dual-Color Sawblade Radar (Result | Mask)", display_img)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_sawblades()
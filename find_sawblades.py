import cv2
import numpy as np
import glob
import os

FRAME_DIR = "frames"

lower_color = np.array([0, 38, 59])
upper_color = np.array([179, 113, 135])


def detect_sawblades():
    files = sorted(glob.glob(os.path.join(FRAME_DIR, "*.png")))
    if not files:
        print("錯誤：資料夾內沒有圖片")
        return

    print("開始追蹤旋轉電鋸...")

    for f in files:
        img = cv2.imread(f)
        h, w = img.shape[:2]

        # 🌟 2. 設定 ROI (Region of Interest)
        # 為了避開下方 UI 按鈕的雜訊，我們只抓取畫面從最上面到 75% 高度的地方
        roi_bottom = int(h * 0.7)
        # 複製一張原圖用來畫畫，並將下半部塗黑以忽略雜訊
        img_draw = img.copy()
        img_detect = img.copy()
        cv2.rectangle(img_detect, (0, roi_bottom), (w, h), (0, 0, 0), -1)

        # 轉換成 HSV 並套用遮罩
        hsv = cv2.cvtColor(img_detect, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # 🌟 3. 尋找輪廓 (Contours)
        # RETR_EXTERNAL 代表我們只找最外層的輪廓，不用管白塊裡面有沒有破洞
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sawblade_positions = []

        for cnt in contours:
            # 1. 計算這個輪廓的「實際面積」
            contour_area = cv2.contourArea(cnt)

            # 過濾掉太小的灰塵雜訊 (可依你的畫面大小微調)
            if contour_area > 50:
                # 2. 取得能包圍這個色塊的「最小正圓形」的中心點與半徑
                (x, y), radius = cv2.minEnclosingCircle(cnt)

                # 避免除以零的錯誤
                if radius == 0:
                    continue

                # 3. 計算這個完美圓形的「理想面積」 (π * r²)
                circle_area = np.pi * (radius**2)

                # 🌟 4. 計算我們的「信心分數」 (0.0 ~ 1.0)
                confidence = contour_area / circle_area

                # 🌟 5. 設定你的信心門檻 (Threshold)
                # 例如：形狀必須有 65% 像一個圓形，我們才承認它是電鋸
                if confidence > 0.65:
                    center = (int(x), int(y))
                    radius = int(radius)

                    # 把找到的座標存起來
                    sawblade_positions.append(center)

                    # 畫出紅色的圓圈和中心點
                    cv2.circle(img_draw, center, radius, (0, 0, 255), 2)

                    # 在旁邊印出座標與「信心分數」
                    text = f"({center[0]},{center[1]}) C:{confidence:.2f}"
                    cv2.putText(
                        img_draw,
                        text,
                        (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        # 顯示找到幾顆電鋸
        cv2.putText(
            img_draw,
            f"Sawblades: {len(sawblade_positions)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # 畫一條黃線，標示我們設定的 ROI 底部界線
        cv2.line(img_draw, (0, roi_bottom), (w, roi_bottom), (0, 255, 255), 2)

        # 顯示結果
        display_img = cv2.resize(img_draw, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow("Sawblade Radar", display_img)

        # 按 'q' 退出，按其他鍵看下一幀 (設為 30 毫秒會像影片一樣播放)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_sawblades()

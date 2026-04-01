import cv2
import numpy as np

# 🌟 換成你 frames 裡有電鋸的圖片路徑
TEST_IMAGE = "frames/frame_0359.png"


def nothing(x):
    pass


def tune_hsv():
    # 讀取圖片並縮小，以免視窗塞爆螢幕
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print(f"找不到圖片: {TEST_IMAGE}")
        return
    img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)

    # 將圖片從 BGR 轉換為 HSV 顏色空間
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 建立一個視窗與滑桿 (Trackbars)
    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H Min", "HSV Tuner", 0, 179, nothing)
    cv2.createTrackbar("S Min", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("V Min", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("H Max", "HSV Tuner", 179, 179, nothing)
    cv2.createTrackbar("S Max", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V Max", "HSV Tuner", 255, 255, nothing)

    print(
        "請調整滑桿，直到右邊的 Mask 畫面中，電鋸變成『純白色』，其他背景變成『純黑色』。"
    )
    print("按 'q' 鍵離開並印出你設定的數值。")

    while True:
        # 讀取滑桿的當前數值
        h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

        # 設定 HSV 的上下限
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # 根據上下限製作遮罩 (Mask)
        # 在範圍內的顏色變成 255 (白)，範圍外的變成 0 (黑)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 把遮罩套用到原圖上看看效果 (非必要，但方便觀察)
        result = cv2.bitwise_and(img, img, mask=mask)

        # 把三張圖拼在一起顯示 (原圖, 黑白遮罩, 濾色結果)
        mask_bgr = cv2.cvtColor(
            mask, cv2.COLOR_GRAY2BGR
        )  # 轉成彩色格式才能跟原圖合併顯示
        stacked = np.hstack((img, mask_bgr, result))

        cv2.imshow("HSV Tuner (Original | Mask | Result)", stacked)

        # 按 'q' 離開迴圈
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n🎯 恭喜！你找到的電鋸顏色範圍是：")
            print(f"lower_color = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_color = np.array([{h_max}, {s_max}, {v_max}])")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    tune_hsv()

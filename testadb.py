import subprocess
import time
import cv2
import numpy as np

DEVICE_ADDRESS = "127.0.0.1:16384"

BTN_LEFT = (140, 1400)
BTN_RIGHT = (320, 1400)
BTN_JUMP = (700, 1400)


def take_action(action):
    """
    接收 AI 輸出的動作代碼並執行
    0: 向左 (長按)
    1: 向右 (長按)
    2: 跳躍/執行 (輕觸)
    """
    duration = 10

    if action == 0:
        x, y = BTN_LEFT
        action_name = "⬅️ 向左"
    elif action == 1:
        x, y = BTN_RIGHT
        action_name = "➡️ 向右"
    elif action == 2:
        x, y = BTN_JUMP
        action_name = "⬆️ 跳躍/執行"
    else:
        print(f"❌ 錯誤：未知的動作代碼 {action}")
        return

    # 發送 ADB 指令
    command = f"adb -s {DEVICE_ADDRESS} shell input swipe {x} {y} {x} {y} {duration}"
    subprocess.run(command, shell=True)
    print(f"👉 已執行: {action_name} (代碼: {action})")


def get_screen():
    """透過 ADB 獲取當前模擬器畫面"""
    command = f"adb -s {DEVICE_ADDRESS} exec-out screencap -p"
    try:
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        image_bytes = pipe.stdout.read()
        if not image_bytes:
            return None
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"截圖失敗: {e}")
        return None


# --- 互動式測試與視覺迴圈 ---
if __name__ == "__main__":
    print("啟動 AI 視覺引擎...")
    print(
        "請點擊彈出的 OpenCV 視窗，然後按下 'a'(左), 'd'(右), 'w'(跳躍) 測試。按 'q' 離開。"
    )

    while True:
        # 1. 獲取畫面
        screen = get_screen()

        if screen is not None:
            # 為了避免手機畫面在電腦螢幕上太大，我們將畫面縮小 50% 顯示
            # (這只是為了顯示方便，不影響 AI 判斷原始解析度)
            screen_resized = cv2.resize(screen, (0, 0), fx=0.5, fy=0.5)

            # 顯示畫面
            cv2.imshow("AI Vision Radar", screen_resized)

        # 2. 等待鍵盤輸入 (1 毫秒延遲，讓 OpenCV 有時間刷新畫面)
        key = cv2.waitKey(1) & 0xFF

        # 3. 根據輸入執行對應動作
        if key == ord("q"):
            print("關閉視覺引擎。")
            break
        elif key == ord("a"):
            take_action(0)
        elif key == ord("d"):
            take_action(1)
        elif key == ord("w"):
            take_action(2)

    cv2.destroyAllWindows()

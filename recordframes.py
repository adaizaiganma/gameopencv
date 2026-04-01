import subprocess
import cv2
import numpy as np
import os
import time

# ==========================================
# 🌟 设定区
# ==========================================
DEVICE_ADDRESS = "127.0.0.1:16384"  # 你的模拟器位址
OUTPUT_DIR = "frames"               # 存放截图的资料夹

def capture_frames_to_disk():
    # 自动建立资料夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 成功建立资料夹: {OUTPUT_DIR}")

    print("📸 开始连续截图！请切换到模拟器画面进行操作。")
    print("👉 想要停止时，请在此视窗按下 'Ctrl + C'，或在预览画面按 'q' 键。")

    frame_count = 0

    try:
        while True:
            start_time = time.time()

            # 1. 透过 ADB 截取无损 PNG
            command = f"adb -s {DEVICE_ADDRESS} exec-out screencap -p"
            pipe = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            bytes_data = pipe.stdout.read()

            if not bytes_data:
                print("⚠️ 截图失败，请检查 ADB 连线是否正常。")
                time.sleep(1)
                continue

            # 2. 将资料转换为 OpenCV 影像格式
            img_array = np.frombuffer(bytes_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 3. 存盘到 frames 资料夹 (档名自动补零，例如 frame_0001.png)
            frame_count += 1
            filename = os.path.join(OUTPUT_DIR, f"frame_{str(frame_count).zfill(4)}.png")
            cv2.imwrite(filename, frame)

            # 4. 计算处理速度 (FPS) 并印出进度
            fps = 1.0 / (time.time() - start_time)
            print(f"✅ 储存 {filename} (FPS: {fps:.1f})")

            # 5. 显示缩小版的预览视窗，让你知道它正在抓什么
            preview = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
            cv2.imshow("ADB Capture Preview", preview)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 侦测到 Ctrl+C，已手动停止截图。")
    finally:
        cv2.destroyAllWindows()
        print(f"🎉 辛苦了！总共截取了 {frame_count} 张超高清原始图片，已存入 '{OUTPUT_DIR}' 资料夹。")

if __name__ == "__main__":
    capture_frames_to_disk()
import scrcpy
import cv2
import time

# 設定你的模擬器位址
DEVICE_ADDRESS = "127.0.0.1:16384"

# 用來計算 FPS 的變數
frame_count = 0
start_time = time.time()

def on_frame(frame: cv2.Mat):
    """每當 Android 產生一個新畫面，這個函式就會以光速被呼叫"""
    global frame_count, start_time
    
    if frame is not None:
        frame_count += 1
        
        # 每秒更新一次 FPS
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"🚀 當前串流速度: {fps:.1f} FPS")
            frame_count = 0
            start_time = time.time()

        # 顯示即時畫面 (由於速度太快，可以稍微縮小顯示)
        display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Scrcpy Super Stream", display_frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    print("啟動 Scrcpy 串流引擎...")
    
    # 初始化 Scrcpy 客戶端
    client = scrcpy.Client(device=DEVICE_ADDRESS)
    
    # 綁定事件：當有新畫面時，交給 on_frame 處理
    client.add_listener(scrcpy.EVENT_FRAME, on_frame)
    
    try:
        # 開始串流 (threaded=True 讓它在背景瘋狂抓圖)
        client.start(threaded=True)
        
        # 讓主程式保持運行，直到你按下 Ctrl+C
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n停止串流。")
    finally:
        client.stop()
        cv2.destroyAllWindows()
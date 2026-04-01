import os
import subprocess


def mp4_to_frames_ffmpeg(video_path, output_folder, fps=None):
    """
    使用 FFmpeg 將 MP4 影片轉換為連續圖片
    :param video_path: 影片檔案路徑 (例如: 'gameplay.mp4')
    :param output_folder: 圖片輸出的資料夾名稱 (例如: 'frames')
    :param fps: (選填) 每秒抽取的幀數。如果不填，則按照影片原本的幀率抽取。
    """
    # 1. 如果輸出資料夾不存在，就自動建立一個
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已建立資料夾: {output_folder}")

    # 2. 設定輸出的檔名格式 (frame_0001.png, frame_0002.png...)
    # %04d 代表用 4 位數字補零
    output_pattern = os.path.join(output_folder, "frame_%04d.png")

    # 3. 組合 FFmpeg 指令
    command = [
        "ffmpeg",
        "-i",
        video_path,  # 輸入檔案
        "-y",  # -y 代表如果圖片已存在，直接覆蓋不詢問
    ]

    # 如果有指定 fps (例如只要每秒抽 10 張來減少資料量)，就加入 -vf 參數
    if fps is not None:
        command.extend(["-vf", f"fps={fps}"])

    # 加入輸出路徑
    command.append(output_pattern)

    print(f"🎬 開始將 '{video_path}' 轉換為圖片...")
    print(f"指令: {' '.join(command)}")

    try:
        # 4. 執行指令
        # stdout=subprocess.DEVNULL 隱藏 FFmpeg 龐大的終端機輸出資訊
        # stderr=subprocess.STDOUT 將錯誤訊息導向標準輸出
        subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True
        )
        print(f"✅ 轉換完成！所有圖片已儲存至 '{output_folder}' 資料夾。")

    except FileNotFoundError:
        print(
            "❌ 錯誤：找不到 FFmpeg！請確認你的電腦已安裝 FFmpeg，並已將其加入系統環境變數 (PATH) 中。"
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ 錯誤：FFmpeg 執行失敗，請確認影片路徑是否正確。")


# --- 測試執行 ---
if __name__ == "__main__":
    # 將這裡換成你的影片檔名"
    VIDEO_FILE = "my_gameplay.mp4"

    # 這是我們上一篇用來測試的資料夾名稱
    TARGET_FOLDER = "frames"

    # 執行轉換 (這裡示範不改變幀率，原始影片有幾幀就抽幾張)
    mp4_to_frames_ffmpeg(VIDEO_FILE, TARGET_FOLDER)

    # 如果你覺得圖片太多，可以加上 fps 參數，例如每秒只抽 15 張：
    # mp4_to_frames_ffmpeg(VIDEO_FILE, TARGET_FOLDER, fps=15)

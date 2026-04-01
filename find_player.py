import cv2
import numpy as np
import glob
import os

# 🌟 1. 設定兩個資料夾的路徑
FRAME_DIR = "frames"  # 遊戲畫面的資料夾
PLAYER_IMG_DIR = "player_img"  # 主角所有模板圖片的資料夾


def run_multi_template_detection():
    templates = []

    # 🌟 2. 自動抓取 player_img 資料夾下所有的 .png 檔案
    # 注意：如果你存的是 .jpg，請改成 "*.jpg"
    template_files = glob.glob(os.path.join(PLAYER_IMG_DIR, "*.png"))

    # 檢查有沒有找到圖片
    if not template_files:
        print(f"❌ 錯誤：在 '{PLAYER_IMG_DIR}' 資料夾中找不到任何圖片！")
        return

    print(
        f"📁 成功在 '{PLAYER_IMG_DIR}' 找到 {len(template_files)} 張模板圖片，開始載入..."
    )

    # 🌟 3. 迴圈載入所有找到的模板
    for f in template_files:
        temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if temp is not None:
            # 為了讓畫面顯示好看一點，我們只取檔名（去掉路徑和副檔名）
            # 例如 'player_img\jump.png' 會變成 'jump'
            base_name = os.path.splitext(os.path.basename(f))[0]
            templates.append((temp, temp.shape[1], temp.shape[0], base_name))
        else:
            print(f"⚠️ 警告：無法讀取圖片 {f}")

    # 取得所有的遊戲幀
    files = sorted(glob.glob(os.path.join(FRAME_DIR, "*.png")))
    if not files:
        print(f"❌ 錯誤：在 '{FRAME_DIR}' 資料夾內沒有遊戲畫面圖片")
        return

    print("🚀 開始自動多模板追蹤測試...")

    for f in files:
        img = cv2.imread(f)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 初始化最佳匹配結果
        best_match = {
            "max_val": -1.0,
            "max_loc": (0, 0),
            "name": "LOST",
            "w": 0,
            "h": 0,
        }

        # 遍歷所有載入的模板進行比對
        for temp_img, temp_w, temp_h, temp_name in templates:
            res = cv2.matchTemplate(gray_img, temp_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # 更新最高分
            if max_val > best_match["max_val"]:
                best_match = {
                    "max_val": max_val,
                    "max_loc": max_loc,
                    "name": temp_name,
                    "w": temp_w,
                    "h": temp_h,
                }

        # 設定門檻值 (可以根據實際情況微調)
        threshold = 0.6

        # 繪製結果
        if best_match["max_val"] >= threshold:
            top_left = best_match["max_loc"]
            bottom_right = (
                top_left[0] + best_match["w"],
                top_left[1] + best_match["h"],
            )
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                img,
                f"Found: {best_match['name']} ({best_match['max_val']:.2f})",
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                img,
                f"LOST (Max: {best_match['max_val']:.2f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        display_img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow("AI Tracking - Auto Folder", display_img)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_multi_template_detection()

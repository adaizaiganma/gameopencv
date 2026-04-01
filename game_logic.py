import math


class GameLogicEngine:
    def __init__(self):
        # 設定地面的 Y 座標 (假設你的遊戲畫面底部 Y 約為 600，請依實際情況修改)
        self.ground_y = 600

        # 碰撞半徑 (像素距離)：主角中心和電鋸中心距離小於多少算死掉
        self.death_radius = 45

        # 用來記錄「正在跳越中」的電鋸 X 座標
        self.dodging_sawblades = []

        self.score = 0

    def check_state(self, player_pos, sawblade_positions):
        """
        傳入當前影格的主角座標與電鋸座標清單，回傳 (是否死亡, 獲得獎勵分數)
        """
        # 如果找不到主角，先當作沒事發生 (或者視為死亡，取決於你的設計)
        if not player_pos:
            return False, 0

        px, py = player_pos
        reward = 0
        is_dead = False

        # ---------------------------------------------------------
        # 規則 1：碰到電鋸就會死掉 (計算歐幾里得距離)
        # ---------------------------------------------------------
        for sx, sy in sawblade_positions:
            # math.hypot 用來計算兩點之間的直線距離 (畢氏定理)
            distance = math.hypot(px - sx, py - sy)

            if distance < self.death_radius:
                print(
                    f"💀 死亡！與電鋸距離 {distance:.1f} < 判定半徑 {self.death_radius}"
                )
                is_dead = True
                return is_dead, -100  # 死掉扣 100 分

        # ---------------------------------------------------------
        # 規則 2：跳起來，經過(X軸重合)，落地後加分
        # ---------------------------------------------------------
        # 判斷主角是否在空中 (Y 數值比地面 Y 數值小一段距離)
        is_in_air = py < (self.ground_y - 20)

        if is_in_air:
            # 主角在空中，檢查有沒有電鋸在他的正下方 (X 軸相近)
            for sx, sy in sawblade_positions:
                # 如果電鋸在主角下方，且 X 軸差距小於 30 像素
                if sy > py and abs(px - sx) < 30:
                    if sx not in self.dodging_sawblades:
                        self.dodging_sawblades.append(sx)
                        print(f"✨ 在空中成功跨越電鋸 (X: {sx})，等待落地...")

        else:
            # 主角在地面
            if len(self.dodging_sawblades) > 0:
                # 剛剛有跨越電鋸，現在落地了！加分！
                earned_points = len(self.dodging_sawblades)
                reward += earned_points * 10  # 每個電鋸加 10 分
                self.score += earned_points
                print(f"💰 成功落地！獲得 {earned_points * 10} 分，總分: {self.score}")

                # 清空紀錄，準備下一次跳躍
                self.dodging_sawblades.clear()

        return is_dead, reward


# --- 簡單的邏輯測試 ---
if __name__ == "__main__":
    engine = GameLogicEngine()
    engine.ground_y = 600

    print("--- 模擬情境 1: 站在地上，電鋸在遠方 ---")
    dead, rew = engine.check_state(
        player_pos=(100, 600), sawblade_positions=[(300, 600)]
    )

    print("\n--- 模擬情境 2: 跳到半空中，電鋸剛好在下方 ---")
    dead, rew = engine.check_state(
        player_pos=(300, 450), sawblade_positions=[(305, 600)]
    )

    print("\n--- 模擬情境 3: 成功落地 ---")
    dead, rew = engine.check_state(
        player_pos=(350, 600), sawblade_positions=[(200, 600)]
    )

    print("\n--- 模擬情境 4: 落地沒踩好，撞到電鋸 ---")
    dead, rew = engine.check_state(
        player_pos=(200, 600), sawblade_positions=[(210, 600)]
    )

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# --- 1. 計算ロジック (オリジナルのランダム配置を維持) ---
class Sandpile:
    def __init__(self, size, threshold=4, start_filled=True):
        self.size = size
        self.threshold = threshold
        if start_filled:
            self.grid = np.random.randint(0, threshold, (size, size))
        else:
            self.grid = np.zeros((size, size), dtype=int)

    def add_grain(self):
        x, y = np.random.randint(0, self.size, size=2)
        self.grid[x, y] += 1
        return self.stabilize_vectorized()

    def stabilize_vectorized(self):
        total_topples = 0
        while np.any(self.grid >= self.threshold):
            over_threshold = self.grid >= self.threshold
            grains_to_move = self.grid // self.threshold
            total_topples += np.sum(over_threshold)
            self.grid -= grains_to_move * self.threshold
            padded = np.pad(grains_to_move, 1, mode='constant')
            self.grid += padded[0:-2, 1:-1] 
            self.grid += padded[2:, 1:-1]   
            self.grid += padded[1:-1, 0:-2] 
            self.grid += padded[1:-1, 2:]   
        return total_topples

# --- 2. UI設定 ---
st.set_page_config(layout="wide")
st.title("3D Bar-Style Sandpile")

# 3D棒グラフは描画負荷が高いため、サイズは30前後が最も滑らかに見えます
size = st.sidebar.slider("Grid Size", 10, 50, 25)
steps = st.sidebar.number_input("Total Steps", 100, 10000, 2000)
update_interval = st.sidebar.select_slider("Update Interval", options=[1, 5, 10, 20, 50], value=10)

if st.button('Start 3D Bar Simulation'):
    model = Sandpile(size=size, start_filled=True)
    image_spot = st.empty()
    
    # 座標の準備
    _x = np.arange(size)
    _y = np.arange(size)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = model.grid.ravel()
    bottom = np.zeros_like(top)
    width = depth = 0.8 # 柱の太さ（1.0にすると隙間がなくなります）

    for step in range(1, steps + 1):
        model.add_grain()

        if step % update_interval == 0:
            # 高さが0より大きい場所だけを抽出して描画を高速化
            z_data = model.grid.ravel()
            mask = z_data > 0
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 3D棒グラフの描画
            # 高さに応じて色を変える（崩壊がわかりやすいようにviridisを使用）
            colors = plt.cm.viridis(z_data[mask] / 4.0)
            ax.bar3d(x[mask], y[mask], bottom[mask], width, depth, z_data[mask], 
                     shade=True, color=colors)

            # 視点と軸の固定
            ax.set_zlim(0, 5)
            ax.view_init(elev=30, azim=45)
            ax.set_axis_off()

            # 画像としてバッファに保存
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=80)
            image_spot.image(buf, use_container_width=True)
            plt.close(fig)

    st.success("Complete")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# --- 1. 計算ロジック ---
class Sandpile:
    def __init__(self, size, threshold=4, start_filled=False):
        self.size = size
        self.threshold = threshold
        if start_filled:
            self.grid = np.random.randint(0, threshold, (size, size))
        else:
            self.grid = np.zeros((size, size), dtype=int)

    def add_grain(self):
        # 中央に砂を落とすと「山ができて崩れる」のが一番よくわかります
        center = self.size // 2
        self.grid[center, center] += 1
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
st.title("3D Sandpile Avalanche Visualizer")

size = st.sidebar.slider("Grid Size", 20, 60, 40)
steps = st.sidebar.number_input("Total Steps", 100, 20000, 5000)
update_interval = st.sidebar.select_slider("Update Interval", options=[1, 10, 20, 50, 100], value=20)

if st.button('Start 3D Visualization'):
    model = Sandpile(size=size, start_filled=False)
    
    # 描画エリア
    image_spot = st.empty()
    
    # 座標の準備
    X, Y = np.meshgrid(range(size), range(size))

    for step in range(1, steps + 1):
        model.add_grain()

        if step % update_interval == 0:
            # --- 3D画像をメモリ上に生成 ---
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # 表面をプロット（antialiased=Falseで描画を高速化）
            surf = ax.plot_surface(X, Y, model.grid, cmap='copper', 
                                   linewidth=0, antialiased=False)
            
            # 見た目の固定（ここが「崩れている感」を出すポイント）
            ax.set_zlim(0, 5)        # 高さを固定
            ax.set_axis_off()        # 余計な軸を消す
            ax.view_init(elev=30, azim=45) # 常にこの角度から見る
            
            # 画像として保存してStreamlitに渡す（点滅防止）
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            image_spot.image(buf, use_container_width=True)
            plt.close(fig) # メモリ解放

    st.success("Simulation Complete")

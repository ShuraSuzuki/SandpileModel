import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# --- 1. 元の計算ロジック (変更なし) ---
class Sandpile:
    def __init__(self, size, threshold=4, start_filled=True):
        self.size = size
        self.threshold = threshold
        if start_filled:
            self.grid = np.random.randint(0, threshold, (size, size))
        else:
            self.grid = np.zeros((size, size), dtype=int)

    def add_grain(self):
        # 元のロジック：ランダムな位置に砂を落とす
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
st.title("3D Sandpile Avalanche Visualizer")

size = st.sidebar.slider("Grid Size", 20, 100, 50)
steps = st.sidebar.number_input("Total Steps", 100, 50000, 20000)
update_interval = st.sidebar.select_slider("Update Interval", options=[1, 10, 50, 100, 200], value=50)

start_mode = st.sidebar.radio("Initial State", ["Randomly Filled", "Empty"], index=0)
is_start_filled = (start_mode == "Randomly Filled")

if st.button('Start Simulation'):
    model = Sandpile(size=size, start_filled=is_start_filled)
    
    # 描画エリア
    image_spot = st.empty()
    
    # 3D座標の準備
    X, Y = np.meshgrid(range(size), range(size))

    for step in range(1, steps + 1):
        model.add_grain()

        if step % update_interval == 0:
            # Python側で3Dグラフを「画像」として作成
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            # antialiased=Falseで描画速度を稼ぐ
            surf = ax.plot_surface(X, Y, model.grid, cmap='viridis', 
                                   linewidth=0, antialiased=False)
            
            # 見た目の固定設定
            ax.set_zlim(0, 5) # 高さを固定して「崩れた」瞬間をわかりやすくする
            ax.view_init(elev=35, azim=45) # 斜め上からの視点に固定
            ax.set_axis_off() # 地図などの余計な情報は出さない
            
            # メモリ上でPNG化（これが点滅防止の鍵）
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            image_spot.image(buf, use_container_width=True)
            plt.close(fig)

    st.success("Simulation Complete")

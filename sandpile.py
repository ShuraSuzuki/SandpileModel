import streamlit as st
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# --- 1. 計算ロジック ---
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
st.title("High-Speed Sandpile Visualizer")

size = st.sidebar.slider("Grid Size", 20, 100, 70)
steps = st.sidebar.number_input("Total Steps", 100, 50000, 20000)
update_interval = st.sidebar.select_slider("Update Interval", options=[1, 10, 50, 100, 200], value=50)

if st.button('Start Simulation'):
    model = Sandpile(size=size, start_filled=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Avalanche Flow (Bright = High)")
        # ここに画像を高速表示する
        state_image = st.empty()
    with col2:
        st.subheader("Avalanche Size Log")
        ts_chart = st.empty()

    avalanche_sizes = []

    for step in range(1, steps + 1):
        topples = model.add_grain()
        avalanche_sizes.append(topples if topples > 0 else 0.1)

        if step % update_interval == 0:
            # --- 爆速・無点滅の秘訣：画像を直接生成して表示 ---
            # gridを0-255のグレースケールに変換し、ヒートマップ化
            # 3（しきい値直前）を一番明るくする
            img_data = (model.grid / 3.0)
            
            # st.imageを使うことで、グラフ描画のオーバーヘッドを無くし点滅を防止
            state_image.image(img_data, clamp=True, use_container_width=True)
            
            # 右側のグラフも更新
            ts_chart.line_chart(avalanche_sizes[-1000:])

    st.success("Simulation Complete")

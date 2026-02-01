import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter

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
st.set_page_config(layout="wide", page_title="Professional Sandpile Sim")
st.title("3D Bar Sandpile with Statistics")

# サイドバー設定
st.sidebar.header("Simulation Settings")
size = st.sidebar.slider("Grid Size", 10, 60, 30)
steps = st.sidebar.number_input("Total Steps", 100, 20000, 5000)
update_interval = st.sidebar.select_slider("Update Interval", options=[1, 5, 10, 20, 50, 100], value=20)

st.sidebar.header("3D View Angle")
elev = st.sidebar.slider("Elevation (仰角)", 0, 90, 30)
azim = st.sidebar.slider("Azimuth (方位角)", -180, 180, 45)

start_mode = st.sidebar.radio("Initial State", ["Randomly Filled", "Empty"], index=0)
is_start_filled = (start_mode == "Randomly Filled")

if st.button('Start Simulation'):
    model = Sandpile(size=size, start_filled=is_start_filled)
    
    # レイアウト: 左側に3D表示、右側にグラフ2つ
    col1, col2 = st.columns([3, 2])
    with col1:
        image_spot = st.empty()
    with col2:
        ts_chart = st.empty()
        dist_plot = st.empty()

    avalanche_sizes = []
    
    # 3D座標の固定データ
    _x = np.arange(size)
    _y = np.arange(size)
    _xx, _yy = np.meshgrid(_x, _y)
    x_flat, y_flat = _xx.ravel(), _yy.ravel()
    width = depth = 0.8

    for step in range(1, steps + 1):
        topples = model.add_grain()
        avalanche_sizes.append(topples if topples > 0 else 0.1)

        if step % update_interval == 0 or step == steps:
            # --- 3D棒グラフ作成 ---
            z_data = model.grid.ravel()
            mask = z_data > 0
            
            fig3d = plt.figure(figsize=(10, 8))
            ax3d = fig3d.add_subplot(111, projection='3d')
            
            if np.any(mask):
                colors = plt.cm.magma(z_data[mask] / 4.0)
                ax3d.bar3d(x_flat[mask], y_flat[mask], np.zeros_like(z_data[mask]), 
                           width, depth, z_data[mask], 
                           shade=True, color=colors)

            ax3d.set_zlim(0, 5)
            ax3d.view_init(elev=elev, azim=azim)
            ax3d.set_axis_off()
            
            buf = BytesIO()
            fig3d.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            image_spot.image(buf, use_container_width=True)
            plt.close(fig3d)

            # --- 右側：時系列グラフ (直近1000件) ---
            ts_chart.line_chart(avalanche_sizes[-1000:], height=250)

            # --- 右側：べき乗則分布 (Log-Log) ---
            valid_sizes = [s for s in avalanche_sizes if s >= 1]
            if valid_sizes:
                counts = Counter(valid_sizes)
                sizes = sorted(counts.keys())
                probs = [counts[s] / len(valid_sizes) for s in sizes]
                
                fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
                ax_dist.scatter(sizes, probs, alpha=0.6, s=20, c='royalblue')
                ax_dist.set_xscale('log')
                ax_dist.set_yscale('log')
                ax_dist.set_xlabel("Avalanche Size (s)")
                ax_dist.set_ylabel("Probability P(s)")
                ax_dist.set_title("Size Distribution")
                ax_dist.grid(True, which="both", ls="-", alpha=0.2)
                
                dist_plot.pyplot(fig_dist)
                plt.close(fig_dist)

    st.success("Simulation Complete")

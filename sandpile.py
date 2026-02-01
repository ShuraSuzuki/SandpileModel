import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. 計算ロジック (高速化のためベクトル化済みのものを継承) ---
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
            self.grid += padded[0:-2, 1:-1] # 上
            self.grid += padded[2:, 1:-1]   # 下
            self.grid += padded[1:-1, 0:-2] # 左
            self.grid += padded[1:-1, 2:]   # 右
        return total_topples

# --- 2. Streamlit UI設定 ---
st.set_page_config(layout="wide", page_title="High-Speed Sandpile")
st.title("Bak-Tang-Wiesenfeld Sandpile Simulation")

# サイドバー設定
size = st.sidebar.slider("Grid Size", 20, 100, 50)
steps = st.sidebar.number_input("Total Steps", 100, 20000, 2000)
update_interval = st.sidebar.select_slider("Update Interval (steps)", options=[10, 50, 100, 200, 500], value=100)

if st.button('Start Simulation'):
    model = Sandpile(size=size)
    
    # レイアウト作成
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Sandpile State")
        state_plot = st.empty()
    with col2:
        st.subheader("Avalanche Statistics")
        ts_plot = st.empty()     # 時系列用
        dist_plot = st.empty()   # べき乗則分布用

    avalanche_sizes = []

    # シミュレーションループ
    for step in range(1, steps + 1):
        topples = model.add_grain()
        avalanche_sizes.append(topples if topples > 0 else 0.1)
        
        # 指定間隔ごとに描画を更新
        if step % update_interval == 0 or step == steps:
            # 1. 砂山の状態 (Matplotlibを使用)
            fig_state, ax_state = plt.subplots(figsize=(5, 5))
            ax_state.imshow(model.grid, cmap='magma', vmin=0, vmax=3)
            ax_state.axis('off')
            state_plot.pyplot(fig_state)
            plt.close(fig_state)
            
            # 2. 時系列グラフ (Streamlitネイティブチャートで高速化)
            # 直近1000件を表示
            ts_plot.line_chart(avalanche_sizes[-1000:], height=200)
            
            # 3. べき乗則分布 (Log-Log)
            valid_sizes = [s for s in avalanche_sizes if s >= 1]
            if valid_sizes:
                counts = Counter(valid_sizes)
                sizes = sorted(counts.keys())
                probs = [counts[s] / len(valid_sizes) for s in sizes]
                
                fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
                ax_dist.scatter(sizes, probs, alpha=0.5, s=15, c='blue')
                ax_dist.set_xscale('log')
                ax_dist.set_yscale('log')
                ax_dist.set_xlabel("Size (s)")
                ax_dist.set_ylabel("P(s)")
                ax_dist.grid(True, which="both", ls="-", alpha=0.2)
                dist_plot.pyplot(fig_dist)
                plt.close(fig_dist)

    st.success(f"Completed {steps} steps!")
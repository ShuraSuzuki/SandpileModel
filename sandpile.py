import streamlit as st
import numpy as np
import plotly.graph_objects as go
from collections import Counter

# --- 1. 計算ロジック (Sandpileクラス) ---
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

# --- 2. Streamlit UI設定 ---
st.set_page_config(layout="wide", page_title="Smooth 3D Sandpile")
st.title("Smooth 3D Sandpile Simulation")

st.sidebar.header("Settings")
# 3Dを滑らかにするため、グリッドサイズは30-40程度がおすすめです
size = st.sidebar.slider("Grid Size", 20, 60, 30)
steps = st.sidebar.number_input("Total Steps", 100, 50000, 20000)
# アニメーションにするステップ数（一度に描画する塊）
frames_to_show = st.sidebar.slider("Animation Frames", 10, 100, 50)
update_interval = 50 # 50ステップごとを1フレームとする

start_mode = st.sidebar.radio("Initial State", ["Randomly Filled", "Empty"], index=0)
is_start_filled = (start_mode == "Randomly Filled")

if st.button('Start Smooth 3D Simulation'):
    model = Sandpile(size=size, start_filled=is_start_filled)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("3D Smooth View")
        view_3d = st.empty()
    with col2:
        st.subheader("Statistics")
        ts_chart = st.empty()
        dist_plot = st.empty()

    avalanche_sizes = []
    
    # 進行状況バー
    progress_bar = st.progress(0)

    # アニメーション用のデータを蓄積するループ
    for f in range(frames_to_show):
        # 指定のステップ数（update_interval）分、計算だけ進める
        for _ in range(update_interval):
            topples = model.add_grain()
            avalanche_sizes.append(topples if topples > 0 else 0.1)

        # --- 3D描画更新（ここを1回にまとめることで点滅を抑える） ---
        fig_3d = go.Figure(data=[go.Surface(z=model.grid, colorscale='Magma')])
        fig_3d.update_layout(
            scene=dict(
                zaxis=dict(range=[0, 4]),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600,
            uirevision='constant' # これが重要：回転状態を維持して点滅を抑える
        )
        view_3d.plotly_chart(fig_3d, use_container_width=True)
        
        # 統計の更新
        ts_chart.line_chart(avalanche_sizes[-1000:], height=200)
        
        # 進捗更新
        progress_bar.progress((f + 1) / frames_to_show)

    st.success("Simulation Batch Complete!")
    
# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import Counter

# # --- 1. 計算ロジック ---
# class Sandpile:
#     def __init__(self, size, threshold=4, start_filled=True):
#         self.size = size
#         self.threshold = threshold
#         if start_filled:
#             self.grid = np.random.randint(0, threshold, (size, size))
#         else:
#             self.grid = np.zeros((size, size), dtype=int)

#     def add_grain(self):
#         x, y = np.random.randint(0, self.size, size=2)
#         self.grid[x, y] += 1
#         return self.stabilize_vectorized()

#     def stabilize_vectorized(self):
#         total_topples = 0
#         while np.any(self.grid >= self.threshold):
#             over_threshold = self.grid >= self.threshold
#             grains_to_move = self.grid // self.threshold
#             total_topples += np.sum(over_threshold)
            
#             self.grid -= grains_to_move * self.threshold
#             padded = np.pad(grains_to_move, 1, mode='constant')
#             self.grid += padded[0:-2, 1:-1] # 上
#             self.grid += padded[2:, 1:-1]   # 下
#             self.grid += padded[1:-1, 0:-2] # 左
#             self.grid += padded[1:-1, 2:]   # 右
#         return total_topples

# # --- 2. Streamlit UI設定 ---
# st.set_page_config(layout="wide", page_title="Custom Sandpile App")
# st.title("Bak-Tang-Wiesenfeld Sandpile Simulation")

# # サイドバー設定
# st.sidebar.header("Simulation Settings")
# size = st.sidebar.slider("Grid Size", 20, 100, 50)

# # 【変更】初期値を 20,000 に設定
# steps = st.sidebar.number_input("Total Steps", 100, 50000, 20000)

# # 【変更】初期値を 50 に設定
# update_interval = st.sidebar.select_slider("Update Interval (steps)", options=[1, 10, 50, 100, 200, 500], value=50)

# start_mode = st.sidebar.radio(
#     "Initial State",
#     ["Randomly Filled (Critical)", "Empty (Zero)"],
#     index=0
# )
# is_start_filled = (start_mode == "Randomly Filled (Critical)")

# if st.button('Start Simulation'):
#     model = Sandpile(size=size, start_filled=is_start_filled)
    
#     col1, col2 = st.columns([1, 1])
#     with col1:
#         st.subheader("Sandpile State")
#         state_plot = st.empty()
#     with col2:
#         st.subheader("Avalanche Statistics")
#         ts_plot = st.empty()
#         dist_plot = st.empty()

#     avalanche_sizes = []

#     for step in range(1, steps + 1):
#         topples = model.add_grain()
#         avalanche_sizes.append(topples if topples > 0 else 0.1)
        
#         if step % update_interval == 0 or step == steps:
#             # 1. 砂山の状態
#             fig_state, ax_state = plt.subplots(figsize=(5, 5))
#             ax_state.imshow(model.grid, cmap='magma', vmin=0, vmax=3)
#             ax_state.axis('off')
#             state_plot.pyplot(fig_state)
#             plt.close(fig_state)
            
#             # 2. 時系列グラフ (直近1000件)
#             ts_plot.line_chart(avalanche_sizes[-1000:], height=200)
            
#             # 3. べき乗則分布 (Log-Log)
#             valid_sizes = [s for s in avalanche_sizes if s >= 1]
#             if valid_sizes:
#                 counts = Counter(valid_sizes)
#                 sizes = sorted(counts.keys())
#                 probs = [counts[s] / len(valid_sizes) for s in sizes]
                
#                 fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
#                 ax_dist.scatter(sizes, probs, alpha=0.5, s=15, c='blue')
#                 ax_dist.set_xscale('log')
#                 ax_dist.set_yscale('log')
#                 ax_dist.set_xlabel("Size (s)")
#                 ax_dist.set_ylabel("P(s)")
#                 ax_dist.grid(True, which="both", ls="-", alpha=0.2)
#                 dist_plot.pyplot(fig_dist)
#                 plt.close(fig_dist)

#     st.success(f"Completed {steps} steps!")

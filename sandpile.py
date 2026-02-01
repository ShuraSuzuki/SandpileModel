import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
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

# --- 2. Streamlit UI ---
st.set_page_config(layout="wide", page_title="3D Sandpile")
st.title("Final 3D Sandpile (Optimized View)")

size = st.sidebar.slider("Grid Size", 20, 100, 50)
steps = st.sidebar.number_input("Total Steps", 100, 50000, 20000)
update_interval = st.sidebar.select_slider("Update Interval", options=[10, 50, 100, 200], value=50)

if st.button('Start Simulation'):
    model = Sandpile(size=size, start_filled=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        view_3d = st.empty()
    with col2:
        ts_chart = st.empty()

    avalanche_sizes = []

    for step in range(1, steps + 1):
        topples = model.add_grain()
        avalanche_sizes.append(topples if topples > 0 else 0.1)

        if step % update_interval == 0:
            # 座標データの作成 (緯度経度ではなく、単純な数値として扱う)
            x, y = np.indices(model.grid.shape)
            # 座標を0付近に集中させる
            df = pd.DataFrame({
                'lon': (x.flatten() - size/2) * 0.01, 
                'lat': (y.flatten() - size/2) * 0.01,
                'z': model.grid.flatten()
            })

            # 3Dレイヤーの設定
            layer = pdk.Layer(
                "ColumnLayer",
                df,
                get_position=['lon', 'lat'],
                get_elevation='z',
                elevation_scale=500, # 座標系に合わせてスケールを大きく調整
                radius=0.5,
                get_fill_color=["z * 60", 100, 200, 200],
                pickable=True,
            )

            # カメラ位置を強制固定 (砂山の真上に配置)
            view_state = pdk.ViewState(
                latitude=0, 
                longitude=0, 
                zoom=12, 
                pitch=45, 
                bearing=0
            )
            
            # 地図を消し、背景を暗くして描画
            view_3d.pydeck_chart(pdk.Deck(
                layers=[layer], 
                initial_view_state=view_state,
                map_style=None, # 背景を完全に無効化
            ))
            
            ts_chart.line_chart(avalanche_sizes[-1000:])

    st.success("Simulation Running...")

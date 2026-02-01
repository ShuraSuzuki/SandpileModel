import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. 計算ロジック部分 (Sandpileクラス) ---
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

# --- 2. Streamlit UI部分 ---
st.title("Self-Organized Criticality: Sandpile Model")

# サイドバーの設定
size = st.sidebar.slider("Grid Size", 20, 100, 50)
steps = st.sidebar.number_input("Total Steps", 100, 10000, 1000)

if st.button('Start Simulation'):
    model = Sandpile(size=size) # ここでクラスを呼び出し
    
    plot_spot = st.empty()
    avalanche_sizes = []

    for step in range(steps):
        topples = model.add_grain()
        avalanche_sizes.append(topples if topples > 0 else 0.1)
        
        # 20ステップごとに画面を更新（速すぎるとブラウザが重くなるため）
        if step % 20 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 左：砂山の状態
            ax1.imshow(model.grid, cmap='magma', vmin=0, vmax=3)
            ax1.set_title(f"Step: {step}")
            
            # 右：べき乗則の分布
            valid_sizes = [s for s in avalanche_sizes if s >= 1]
            if valid_sizes:
                counts = Counter(valid_sizes)
                ax2.scatter(counts.keys(), [v/len(valid_sizes) for v in counts.values()], alpha=0.5)
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax2.set_xlabel("Avalanche Size")
                ax2.set_ylabel("Probability")
                ax2.set_title("Size Distribution (Log-Log)")

            plot_spot.pyplot(fig)
            plt.close(fig)
    
    st.success("Simulation Complete!")
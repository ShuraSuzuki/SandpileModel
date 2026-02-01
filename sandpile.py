import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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

# --- Streamlit UI ---
st.set_page_config(layout="wide") # 画面を広く使う
st.title("Bak-Tang-Wiesenfeld Sandpile Model")

size = st.sidebar.slider("Grid Size", 20, 100, 50)
steps = st.sidebar.number_input("Total Steps", 100, 10000, 1000)

if st.button('Start Simulation'):
    model = Sandpile(size=size)
    plot_spot = st.empty()
    avalanche_sizes = []

    for step in range(steps):
        topples = model.add_grain()
        # ログスケールのために0は0.1として扱う
        avalanche_sizes.append(topples if topples > 0 else 0.1)
        
        if step % 25 == 0:
            # 3つのグラフを横に並べる
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            # 1. 砂山の状態
            ax1.imshow(model.grid, cmap='magma', vmin=0, vmax=3)
            ax1.set_title(f"Sandpile State (Step: {step})")
            
            # 2. 時系列グラフ (ステップ数 vs 雪崩規模)
            ax2.plot(avalanche_sizes, color='red', lw=0.5)
            ax2.set_yscale('log')
            ax2.set_title("Avalanche Size (Time Series)")
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Size")
            
            # 3. べき乗則の分布 (Log-Log)
            valid_sizes = [s for s in avalanche_sizes if s >= 1]
            if valid_sizes:
                counts = Counter(valid_sizes)
                sizes = sorted(counts.keys())
                probs = [counts[s] / len(valid_sizes) for s in sizes]
                ax3.scatter(sizes, probs, alpha=0.5, s=10)
                ax3.set_xscale('log')
                ax3.set_yscale('log')
                ax3.set_title("Size Distribution (Log-Log)")
                ax3.set_xlabel("Size (s)")
                ax3.set_ylabel("P(s)")

            plt.tight_layout()
            plot_spot.pyplot(fig)
            plt.close(fig)

    st.success("Simulation Complete!")
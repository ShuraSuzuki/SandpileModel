import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

# --- Sandpileクラスはそのまま使用 ---
class Sandpile:
    # (お手持ちのコードをここにコピー)
    ...

# --- Streamlit UI ---
st.title("Self-Organized Criticality: Sandpile Model")

# サイドバーでパラメータ設定
size = st.sidebar.slider("Grid Size", 20, 100, 50)
steps = st.sidebar.number_input("Total Steps", 100, 10000, 1000)

if st.button('Start Simulation'):
    model = Sandpile(size)
    
    # 表示用のプレースホルダ
    status_text = st.empty()
    plot_spot = st.empty()
    
    avalanche_sizes = []

    for step in range(steps):
        topples = model.add_grain()
        avalanche_sizes.append(topples if topples > 0 else 0.1)
        
        # 定期的に描画を更新
        if step % 10 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 左：砂山の状態
            ax1.imshow(model.grid, cmap='magma')
            ax1.set_title(f"Step: {step}")
            
            # 右：べき乗則の分布
            valid_sizes = [s for s in avalanche_sizes if s >= 1]
            if valid_sizes:
                counts = Counter(valid_sizes)
                ax2.scatter(counts.keys(), [v/len(valid_sizes) for v in counts.values()], alpha=0.5)
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax2.set_title("Size Distribution (Log-Log)")

            plot_spot.pyplot(fig)
            plt.close(fig) # メモリ節約
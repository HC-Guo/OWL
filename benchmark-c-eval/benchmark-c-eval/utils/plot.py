import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'Heiti TC'
def draw_radar_chart(values,categories,name,draw_path):
    # 角度计算
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # 数据重复首尾，以闭合雷达图
    values += values[:1]

    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})

    # 绘制雷达图
    ax.plot(angles, values, color='b', linewidth=1)
    ax.fill(angles, values, color='b', alpha=0.25)

    # 设置标题和刻度标签
    ax.set_title(name, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # 设置刻度范围
    ax.set_ylim(0, max(values) + 1)

    plt.legend()
    plt.savefig(draw_path)

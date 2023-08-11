import numpy as np
import matplotlib.pyplot as plt

# 数据
categories = ['A', 'B', 'C', 'D', 'E']  # 各个类别
values = [4, 3, 5, 2, 1]  # 对应各个类别的数值

# 角度计算
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# 数据重复首尾，以闭合雷达图[ 'B','A', 'C', 'D', 'E']
values += values[:1]

# 创建画布和子图
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'polar': True})

# 绘制雷达图
ax.plot(angles, values, color='b', linewidth=1,label="vv")
ax.fill(angles, values, color='b', alpha=0.25)

v=[4, 3, 6, 3, 4,4]
ax.plot(angles, v, color='b', linewidth=1,label="a")
ax.fill(angles, v, color='r', alpha=0.25)
# 设置标题和刻度标签
ax.set_title('Radar Chart', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# 设置刻度范围
ax.set_ylim(0, max(values) + 0.5)

plt.legend()
plt.show()

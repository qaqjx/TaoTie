import matplotlib.pyplot as plt

# 创建图形
plt.figure(figsize=(6, 6))

# 设置x轴和y轴的范围，只显示第一象限
plt.xlim(0, 10)
plt.ylim(0, 10)

# 绘制x轴和y轴
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# 添加箭头
plt.annotate('', xy=(10, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='black', linewidth=1))
plt.annotate('', xy=(0, 10), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='black', linewidth=1))

# 添加x轴和y轴标签
plt.xlabel('Time')
plt.ylabel('Quality')

# 隐藏边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# 隐藏刻度线
plt.xticks([])
plt.yticks([])

# 示例数据点和形状
points = [(2, 7), (2.2, 8.8), (8, 9)]
markers = ['o', 's', '^']  # 圆形、方形、三角形
names = ['RAG', 'Ours', 'Long Context']

# 绘制每个点并添加名称
for point, marker, name in zip(points, markers, names):
    plt.scatter(point[0], point[1], marker=marker, s=100)  # s=100 设置点的大小
    plt.text(point[0] + 0.2, point[1], name, fontsize=9)  # 添加名称，稍微偏移以避免重叠

# 保存图形到文件
plt.savefig('first_quadrant_with_arrows_and_points.png')

# 显示图形
plt.show()
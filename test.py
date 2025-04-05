import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 创建一个虚拟数据集
np.random.seed(0)
X = np.random.rand(1000, 50)  # 1000个样本，每个样本50个特征

# 应用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
tsne_2d = tsne.fit_transform(X)

# 为每个点分配一个随机标签
labels = np.random.randint(0, 10, 1000)

# 创建图形和轴
fig3, ax3 = plt.subplots(figsize=(10, 7))

# 绘制散点图
sc_all = ax3.scatter(
    tsne_2d[:, 0],
    tsne_2d[:, 1],
    c=labels,
    cmap='tab10',
    s=10,  # 增加点的大小以提高可见性
    marker='o',
    alpha=0.6,  # 增加透明度以提高重叠点的可见性
    label='All Points (After)'
)

# 添加图例
ax3.legend()

# 使用 x 和 y 轴分别设置刻度线的方向
ax3.xaxis.set_tick_params(direction='in', length=6)  # x 轴
ax3.yaxis.set_tick_params(direction='in', length=6)  # y 轴

# 手动调整坐标轴框架
for spine in ax3.spines.values():
    spine.set_linewidth(1)  # 设置坐标轴边框宽度
    spine.set_color('black')  # 设置颜色

# 调整布局
fig3.tight_layout()



# 显示图形
plt.show()
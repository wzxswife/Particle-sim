import os
import numpy as np
import matplotlib.pyplot as plt

# --- 参数设置 ---
L = 100  # 区域大小
N_grid = 100  # 网格数量 (20x20)
dx = L / N_grid  # 网格间距 (5.0)
N_particles = 1000000  # 粒子总数

np.random.seed(42)  # 固定随机种子以保证结果可复现
x_p = np.random.uniform(0, L, N_particles)
y_p = np.random.uniform(0, L, N_particles)

# --- 第一题：粒子数密度统计 ---


def calculate_density_ngp(x, y, n_grid, delta_x):
    """最近网格法 (Nearest Grid Point)"""
    density = np.zeros((n_grid + 1, n_grid + 1))
    # 计算最近的网格点索引
    i = np.round(x / delta_x).astype(int)
    j = np.round(y / delta_x).astype(int)

    # 统计落在每个网格点上的粒子数
    for k in range(len(x)):
        density[j[k], i[k]] += 1
    return density


def calculate_density_first_order(x, y, n_grid, delta_x):
    """一阶权重法 (Area Weighting / Bilinear)"""
    density = np.zeros((n_grid + 1, n_grid + 1))

    # 计算左下角网格索引
    i = (x / delta_x).astype(int)
    j = (y / delta_x).astype(int)

    # 计算权重因子
    dx1 = (x / delta_x) - i
    dy1 = (y / delta_x) - j

    # 分配粒子到四个相邻网格点
    for k in range(len(x)):
        density[j[k], i[k]] += (1 - dx1[k]) * (1 - dy1[k])
        density[j[k], i[k] + 1] += dx1[k] * (1 - dy1[k])
        density[j[k] + 1, i[k]] += (1 - dx1[k]) * dy1[k]
        density[j[k] + 1, i[k] + 1] += dx1[k] * dy1[k]
    return density


# 计算密度
rho_ngp = calculate_density_ngp(x_p, y_p, N_grid, dx)
rho_first = calculate_density_first_order(x_p, y_p, N_grid, dx)

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(rho_ngp, origin="lower", extent=[0, L, 0, L], cmap="Blues")
plt.colorbar(label="n")
plt.title("Nearest Grid Point (NGP)")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.imshow(rho_first, origin="lower", extent=[0, L, 0, L], cmap="Blues")
plt.colorbar(label="n")
plt.title("First-order Weighting")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
pic_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Pic")
os.makedirs(pic_dir, exist_ok=True)
plt.savefig(os.path.join(pic_dir, "density_plot.png"), dpi=300)
print(f"第一题图片已保存为 {pic_dir}/density_plot.png")

# --- 第二题：电场双线性插值 ---


def get_field_at_pos(x, y, delta_x):
    """一阶权重法（双线性插值）求电场 E = x + y²"""
    # 1. 找到粒子所在的网格左下角索引
    i = int(x / delta_x)
    j = int(y / delta_x)

    # 2. 计算相对坐标权重 (0~1)
    u = (x / delta_x) - i
    v = (y / delta_x) - j

    # 3. 计算四个网格点的实际坐标
    x00 = i * delta_x
    y00 = j * delta_x
    x10 = (i + 1) * delta_x
    y10 = j * delta_x
    x01 = i * delta_x
    y01 = (j + 1) * delta_x
    x11 = (i + 1) * delta_x
    y11 = (j + 1) * delta_x

    # 4. 计算四个网格点的场值 E = x + y²
    e00 = x00 + y00 ** 2
    e10 = x10 + y10 ** 2
    e01 = x01 + y01 ** 2
    e11 = x11 + y11 ** 2

    # 5. 双线性插值公式
    ex = (1 - u) * (1 - v) * e00 + u * (1 - v) * e10 + (1 - u) * v * e01 + u * v * e11
    return ex


# 测试点
test_points = [(7.8, 9.5)]

print("--- 第二题：电场插值结果 (E = x + y²) ---")
for px, py in test_points:
    ex_val = get_field_at_pos(px, py, dx)
    # 计算精确值用于对比
    exact_val = px + py ** 2
    print(f"粒子位置 ({px}, {py}) :")
    print(f"  一阶权重法插值结果: E = {ex_val:.4f}")
    print(f"  精确值 (E = x + y²): E = {exact_val:.4f}")
    print(f"  误差: {abs(ex_val - exact_val):.4f}")

# 绘制电场插值图
# 创建网格用于绘制电场分布
x_grid = np.linspace(0, L, 100)
y_grid = np.linspace(0, L, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# 计算每个网格点的电场值
Ex = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Ex[i, j] = get_field_at_pos(X[i, j], Y[i, j], dx)

# 绘制电场热力图
plt.figure(figsize=(8, 6))
plt.imshow(Ex, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r", aspect="equal")
plt.colorbar(label="Ex")
plt.title("Electric Field Interpolation (Ex)")
plt.xlabel("x")
plt.ylabel("y")

# 标记测试点
for px, py in test_points:
    plt.plot(px, py, "ko", markersize=8)
    ex_val = get_field_at_pos(px, py, dx)
    plt.annotate(
        f"Ex={ex_val:.1f}",
        (px, py),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, "electric_field_plot.png"), dpi=300)
print(f"第二题图片已保存为 {pic_dir}/electric_field_plot.png")

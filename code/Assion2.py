import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ================== 参数设置 ==================
x_min, x_max = 0.0, 100.0
N = 101                     # 网格点数
dx = (x_max - x_min) / (N - 1)   # = 1.0
x_grid = np.linspace(x_min, x_max, N)

# 电荷参数
q1, x1 = 1.0, 23.6
q2, x2 = -1.0, 63.3
epsilon0 = 1.0              # 真空介电常数

# ================== 1. 一阶权重法分配电荷到网格 ==================
# 初始化网格电荷量 Q (每个网格点代表长度为 dx 的区间内的总电荷)
Q = np.zeros(N)

def deposit_charge(q, x_pos):
    """一维一阶权重法 (CIC) 将点电荷分配到相邻网格点上"""
    i = int(np.floor(x_pos / dx))          # 左侧网格索引
    if i < 0 or i >= N-1:
        return                              # 超出范围则忽略（本题目在内部）
    w_left = 1.0 - (x_pos - i*dx) / dx     # 左侧权重
    w_right = (x_pos - i*dx) / dx          # 右侧权重
    Q[i] += q * w_left
    Q[i+1] += q * w_right

deposit_charge(q1, x1)
deposit_charge(q2, x2)

# 注意：泊松方程离散形式为 (φ_{i-1} - 2φ_i + φ_{i+1})/dx^2 = -ρ_i/ε0
# 其中 ρ_i = Q_i / dx，因此右边 = -Q_i/(dx * ε0) * dx^2? 仔细推导：
# 离散： (φ_{i-1} - 2φ_i + φ_{i+1})/dx^2 = - (Q_i/dx)/ε0
# => φ_{i-1} - 2φ_i + φ_{i+1} = - (dx * Q_i) / ε0
# 取 dx=1, ε0=1，简化为 φ_{i-1} - 2φ_i + φ_{i+1} = -Q_i
# 因此源项直接用 Q_i 即可。
rhs = -Q  # 方程右边

# ================== 2. 方法一：有限差分法（三对角，Dirichlet边界 φ=0） ==================
# 构建三对角矩阵 A * φ = rhs
# 内点方程：φ_{i-1} - 2φ_i + φ_{i+1} = rhs_i
# 边界条件：φ[0] = 0, φ[N-1] = 0
N_int = N - 2                # 内点个数
# 三对角矩阵元素：主对角线全为 -2，次对角线全为 1
# 使用 banded 格式存储（3 行，分别对应下对角线、主对角线、上对角线）
A_band = np.zeros((3, N))
A_band[0, 1:] = 1.0          # 下对角线
A_band[1, :] = -2.0          # 主对角线
A_band[2, :-1] = 1.0         # 上对角线
# 修改边界条件对应的方程
# 对于 i=0 的方程，φ[0] 已知，实际不参与求解，但我们只求解内点，因此可移除边界点。
# 简便方法：直接构建 N×N 矩阵，但用 banded 需要处理边界。这里采用简单方式：移除边界点。
# 内点索引 1 到 N-2，共 N-2 个未知量。
rhs_int = rhs[1:-1]
# 调整边界影响：对 i=1 的方程，φ[0] 项移到右边；对 i=N-2 的方程，φ[N-1] 项移到右边。
rhs_int[0] -= 0.0            # φ[0]=0，无贡献
rhs_int[-1] -= 0.0           # φ[N-1]=0，无贡献
# 构建内点三对角矩阵
diag = -2.0 * np.ones(N_int)
off_diag = 1.0 * np.ones(N_int-1)
A_tri = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
phi_dir = np.zeros(N)
phi_dir[1:-1] = np.linalg.solve(A_tri, rhs_int)

# 计算电场 (中心差分)
E_dir = np.zeros(N)
E_dir[1:-1] = -(phi_dir[2:] - phi_dir[:-2]) / (2*dx)
# 边界点采用前向/后向差分
E_dir[0] = -(phi_dir[1] - phi_dir[0]) / dx
E_dir[-1] = -(phi_dir[-1] - phi_dir[-2]) / dx

# ================== 3. 方法二：FFT 法（周期边界条件） ==================
# 周期边界条件：φ(0) = φ(L)，要求平均电势为零。
# 傅里叶空间中： -k^2 φ_k = ρ_k  => φ_k = -ρ_k / k^2 (k≠0), φ_0=0
# 离散波数：k_n = 2π n / L, L = N*dx = 100
L = x_max - x_min   # = 100
k_vals = 2 * np.pi * np.fft.fftfreq(N, d=dx)  # 波数数组
rho_k = np.fft.fft(rhs)      # 注意 rhs = -Q，实际是 -ρ_k? 我们直接使用 rhs
phi_k = np.zeros(N, dtype=complex)
# 对 k=0 设为零（平均电势为零）
phi_k[0] = 0.0
# 对 k≠0 求解
non_zero = (k_vals != 0)
phi_k[non_zero] = -rho_k[non_zero] / (k_vals[non_zero]**2)
phi_per = np.fft.ifft(phi_k).real

# 计算电场
E_per = np.zeros(N)
E_per[1:-1] = -(phi_per[2:] - phi_per[:-2]) / (2*dx)
E_per[0] = -(phi_per[1] - phi_per[0]) / dx
E_per[-1] = -(phi_per[-1] - phi_per[-2]) / dx

# ================== 4. 库仑定律直接计算电场（三维真实点电荷） ==================
def coulomb_field(x, q, x0):
    """点电荷 q 在位置 x0 处，在观测点 x 产生的电场（沿 x 方向）"""
    r = x - x0
    return q / (4 * np.pi * epsilon0) * r / (np.abs(r)**3 + 1e-12)  # 加小量避免除零

E_coul = np.zeros(N)
for i, xi in enumerate(x_grid):
    E_coul[i] = coulomb_field(xi, q1, x1) + coulomb_field(xi, q2, x2)

# ================== 5. 结果比较与绘图 ==================
# 计算误差
err_dir = np.abs(E_dir - E_coul)
err_per = np.abs(E_per - E_coul)
max_err_dir = np.max(err_dir)
max_err_per = np.max(err_per)
print(f"有限差分法 (Dirichlet) 最大电场误差: {max_err_dir:.4e}")
print(f"FFT法 (周期边界) 最大电场误差: {max_err_per:.4e}")

# 绘图
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(x_grid, Q, 'o-', label='Grid Charge Q (after deposition)')
plt.xlabel('x')
plt.ylabel('Q')
plt.title('Charge Distribution via First-order Weighting')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x_grid, phi_dir, label='Finite Difference (Dirichlet)')
plt.plot(x_grid, phi_per, '--', label='FFT (Periodic Boundary)')
plt.xlabel('x')
plt.ylabel('Potential $\phi$')
plt.title('Potential Distribution Comparison')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x_grid, E_coul, 'k-', linewidth=2, label='Coulomb Law (Analytic)')
plt.plot(x_grid, E_dir, 'r--', label='Finite Difference E-field')
plt.plot(x_grid, E_per, 'b:', label='FFT E-field')
plt.xlabel('x')
plt.ylabel('Electric Field E')
plt.title('Electric Field Distribution Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('field_comparison.png', dpi=150)
plt.show()

# 输出一些统计信息
print("\n电场偏差统计:")
print(f"有限差分法: 平均绝对误差 = {np.mean(err_dir):.4e}, 最大绝对误差 = {max_err_dir:.4e}")
print(f"FFT法:      平均绝对误差 = {np.mean(err_per):.4e}, 最大绝对误差 = {max_err_per:.4e}")
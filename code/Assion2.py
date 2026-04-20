import numpy as np
import matplotlib.pyplot as plt

# ================== 参数设置 ==================
x_min, x_max = 0.0, 100.0
N = 101                     # 网格点数
dx = (x_max - x_min) / (N - 1)   # = 1.0
x_grid = np.linspace(x_min, x_max, N)

# 电荷参数
q1, x1 = 1.0, 23.6
q2, x2 = -1.0, 63.3
epsilon0 = 1.0              # 真空介电常数


# ================== 电荷分配：一阶权重法 (CIC) ==================
def deposit_charge(q, x_pos, dx, N):
    """
    一维一阶权重法 (CIC) 将点电荷分配到相邻网格点上
    
    Args:
        q: 电荷量
        x_pos: 电荷位置
        dx: 网格步长
        N: 网格点数
    
    Returns:
        Q: 网格电荷量数组
    """
    Q = np.zeros(N)
    i = int(np.floor(x_pos / dx))          # 左侧网格索引
    if i < 0 or i >= N-1:
        return Q                          # 超出范围则忽略
    w_left = 1.0 - (x_pos - i*dx) / dx    # 左侧权重
    w_right = (x_pos - i*dx) / dx         # 右侧权重
    Q[i] += q * w_left
    Q[i+1] += q * w_right
    return Q


# ================== 方法一：有限差分法（Dirichlet边界） ==================
def poisson_direct(rho, dx, phi0, phiN):
    """
    一维泊松方程直接求解（Dirichlet边界，Thomas算法）
    
    离散方程：φ_{i-1} - 2φ_i + φ_{i+1} = -dx²ρ_i
    
    Args:
        rho: array, 网格上的电荷密度 ρ[i]
        dx: float, 网格步长
        phi0: float, 左边界电势 φ(0)
        phiN: float, 右边界电势 φ(L)
    
    Returns:
        phi: array, 电势分布
    """
    N = len(rho)
    
    # 右端项: -dx² * ρ_i
    d = -dx**2 * rho.copy().astype(float)
    
    # 边界条件修正（内部点 1..N-2）
    d[1]   -= phi0
    d[N-2] -= phiN
    
    # 三对角系数
    a = np.ones(N-2)      # 下对角线
    b = -2 * np.ones(N-2) # 主对角线
    c = np.ones(N-2)       # 上对角线
    
    # Thomas 算法
    w = np.zeros(N-2)
    g = np.zeros(N-2)
    
    # 向前消元
    w[0] = c[0] / b[0]
    g[0] = d[1] / b[0]          # 注意 d 索引对应内部点（原始网格索引1）
    for i in range(1, N-2):
        denom = b[i] - a[i] * w[i-1]
        w[i] = c[i] / denom
        g[i] = (d[i+1] - a[i] * g[i-1]) / denom
    
    # 向后回代
    phi_int = np.zeros(N-2)     # 内部点电势
    phi_int[-1] = g[-1]
    for i in range(N-4, -1, -1):
        phi_int[i] = g[i] - w[i] * phi_int[i+1]
    
    # 组合边界值
    phi = np.zeros(N)
    phi[0] = phi0
    phi[1:-1] = phi_int
    phi[-1] = phiN
    return phi


# ================== 方法二：FFT法（周期边界条件） ==================
def poisson_fft_periodic(rho, dx):
    """
    周期边界条件泊松方程求解（平均电势为零）
    
    傅里叶空间中：φ_k = ρ_k / k² (k≠0)，φ_0 = 0
    
    Args:
        rho: array, 电荷密度（长度 N）
        dx: float, 网格步长
    
    Returns:
        phi: array, 电势分布
    """
    N = len(rho)
    L = N * dx
    
    # 计算傅里叶变换
    rho_hat = np.fft.fft(rho)
    
    # 波数数组
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    
    # 避免除以零
    phi_hat = np.zeros(N, dtype=complex)
    
    # n=0 分量置零（平均电势为零）
    phi_hat[0] = 0.0
    
    # 非零波数求解: φ_k = -ρ_k / k²
    mask = (k != 0)
    phi_hat[mask] = rho_hat[mask] / (k[mask]**2)
    
    # 逆变换取实部
    phi = np.fft.ifft(phi_hat).real
    return phi


# ================== 电场计算（中心差分） ==================
def compute_electric_field(phi, dx):
    """
    通过电势计算电场: E = -dφ/dx
    
    Args:
        phi: array, 电势分布
        dx: float, 网格步长
    
    Returns:
        E: array, 电场分布
    """
    N = len(phi)
    E = np.zeros(N)
    
    # 中心差分（内部点）
    E[1:-1] = -(phi[2:] - phi[:-2]) / (2*dx)
    
    # 边界点采用前向/后向差分
    E[0] = -(phi[1] - phi[0]) / dx
    E[-1] = -(phi[-1] - phi[-2]) / dx
    
    return E


# ================== 库仑定律解析解 ==================
def coulomb_field(x, q, x0, epsilon0=1.0):
    """
    点电荷 q 在位置 x0 处产生的电场（沿 x 方向）
    
    E = q / (4πε₀) * r / |r|³
    
    Args:
        x: 观测点位置
        q: 电荷量
        x0: 电荷位置
        epsilon0: 真空介电常数
    
    Returns:
        E: 电场值
    """
    r = x - x0
    # np.sign(0) 为 0，正好处理了电荷中心位置
    return (q / (2.0 * epsilon0)) * np.sign(r)


# ================== 主程序 ==================
if __name__ == "__main__":
    # 电荷分配
    Q1 = deposit_charge(q1, x1, dx, N)
    Q2 = deposit_charge(q2, x2, dx, N)
    Q = Q1 + Q2
    
    # 泊松方程离散: φ_{i-1} - 2φ_i + φ_{i+1} = -Q_i (dx=1, ε₀=1)
    rhs = Q
    
    # 方法1: 有限差分法 (Dirichlet边界 φ=0)
    phi_dir = poisson_direct(rhs, dx, 0.0, 0.0)
    E_dir = compute_electric_field(phi_dir, dx)
    
    # 方法2: FFT法 (周期边界)
    phi_fft = poisson_fft_periodic(rhs, dx)
    E_fft = compute_electric_field(phi_fft, dx)

    E_dir = E_dir - np.mean(E_dir)
    E_fft = E_fft - np.mean(E_fft)
    
    # 库仑定律解析解
    E_coul = np.zeros(N)
    for i, xi in enumerate(x_grid):
        E_coul[i] = coulomb_field(xi, q1, x1, epsilon0) + coulomb_field(xi, q2, x2, epsilon0)
    
    # 误差分析
    err_dir = np.abs(E_dir - E_coul)
    err_fft = np.abs(E_fft - E_coul)
    max_err_dir = np.max(err_dir)
    max_err_fft = np.max(err_fft)
    E_coul_mean = E_coul - np.mean(E_coul) 
    
    print("=" * 50)
    print("电场计算结果对比")
    print("=" * 50)
    print(f"有限差分法 (Dirichlet) 最大电场误差: {max_err_dir:.4e}")
    print(f"FFT法 (周期边界) 最大电场误差:     {max_err_fft:.4e}")
    print()
    print("电场偏差统计:")
    print(f"  有限差分法: 平均误差 = {np.mean(err_dir):.4e}, 最大误差 = {max_err_dir:.4e}")
    print(f"  FFT法:      平均误差 = {np.mean(err_fft):.4e}, 最大误差 = {max_err_fft:.4e}")
    
    # ========== 图1：电荷分布 ==========
    plt.figure(figsize=(14, 4))
    plt.plot(x_grid, Q, 'o-', markersize=5, linewidth=1.5, label='Grid Charge Q (CIC)')
    plt.axvline(x1, color='r', linestyle='--', alpha=0.7, label=f'$q_1=+1$ at $x={x1}$')
    plt.axvline(x2, color='g', linestyle='--', alpha=0.7, label=f'$q_2=-1$ at $x={x2}$')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Q', fontsize=12)
    plt.title('Charge Distribution via First-order Weighting (CIC)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Pic/charge_distribution.png', dpi=150)
    print("图片1已保存为 Pic/charge_distribution.png")
    plt.close()

    # ========== 图2：电势分布 ==========
    plt.figure(figsize=(14, 4))
    plt.plot(x_grid, phi_dir, 'b-', linewidth=2, label='Finite Difference (Dirichlet BC)')
    plt.plot(x_grid, phi_fft, 'r--', linewidth=2, label='FFT (Periodic Boundary)')
    plt.xlabel('x', fontsize=12)
    plt.ylabel(r'Potential $\phi$', fontsize=12)
    plt.title('Electric Potential Distribution Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Pic/potential_distribution.png', dpi=150)
    print("图片2已保存为 Pic/potential_distribution.png")
    plt.close()

    # ========== 图3：电场分布 ==========
    fig, axs = plt.subplots(2, 1, figsize=(16, 8)) 
    # plt.figure(figsize=(14, 4))
    axs[0].plot(x_grid, E_coul, 'k-', linewidth=2.5, label='Coulomb Law (Analytic)')
    axs[0].plot(x_grid, E_dir, 'b--', linewidth=1.5, label='Finite Difference E-field')
    axs[0].plot(x_grid, E_fft, 'r:', linewidth=1.5, label='FFT E-field')
    axs[0].axvline(x1, color='gray', linestyle=':', alpha=0.5)
    axs[0].axvline(x2, color='gray', linestyle=':', alpha=0.5)
    axs[0].set_xlabel('x', fontsize=12)
    axs[0].set_ylabel('Electric Field E', fontsize=12)
    axs[0].set_title('Electric Field Distribution Comparison', fontsize=14)
    axs[0].legend(fontsize=10)
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(x_grid, E_coul_mean, 'k-', linewidth=2.5, label='Coulomb Law (Analytic)')
    axs[1].plot(x_grid, E_dir, 'b--', linewidth=1.5, label='Finite Difference E-field')
    axs[1].plot(x_grid, E_fft, 'r:', linewidth=1.5, label='FFT E-field')
    axs[1].axvline(x1, color='gray', linestyle=':', alpha=0.5)
    axs[1].axvline(x2, color='gray', linestyle=':', alpha=0.5)
    axs[1].set_xlabel('x', fontsize=12)
    axs[1].set_ylabel('Electric Field E', fontsize=12)
    axs[1].set_title('Electric Field Distribution Comparison', fontsize=14)
    axs[1].legend(fontsize=10)
    axs[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Pic/electric_field_distribution.png', dpi=150)
    print("图片3已保存为 Pic/electric_field_distribution.png")
    plt.close()

    print("\n所有图片已保存完成！")

import numpy as np

def acdtw(
    P: np.ndarray,          # 测试序列特征, shape (m, d)
    Q: np.ndarray,          # 模板序列特征, shape (n, d)
    dist_func=None,         # 距离函数 med(i,j) -> float，默认欧氏距离
    window: int = None,     # 可选全局窗口约束 (Sakoe-Chiba band)
) -> tuple:
    """
    计算 ACDTW 最优路径。

    论文中 (i,j) 更新规则:
      ACDTW(1,1) = MED(1,1)
      ACDTW(i,j) = MED(i,j) + min{
          ACDTW(i-1, j-1),
          ACDTW(i-1, j) + C(x_{i-1,j})·MED(i,j),
          ACDTW(i, j-1) + C(x_{i,j-1})·MED(i,j)
      }

    其中 C(x_{i,j}) = 2·max(m,n) / (m+n) · N(x_{i,j})
    N(x_{i,j}) = Pi,j + Qi,j，即两个序列中该帧被使用的累计次数。

    Parameters
    ----------
    P : (m, d) 测试特征
    Q : (n, d) 模板特征
    dist_func : callable(p_vec, q_vec) -> float
        用于计算两帧间距离的函数，默认为欧氏距离
    window : int, optional
        全局窗口约束，若为 None 则不启用

    Returns
    -------
    best_path : np.ndarray (L, 2)
        L 对对应索引 (i, j)，i 属于测试帧索引，j 属于模板帧索引
    med_matrix : np.ndarray (m, n)
        每对帧的距离矩阵
    P_count, Q_count : np.ndarray (m, n)
        每个位置 (i,j) 被路径经过时，对应帧的累计使用次数（用于调试/可视化）
    """
    m, n = len(P), len(Q)
    if window is not None:
        window = max(window, abs(m - n))
    
    if dist_func is None:
        # 默认欧氏距离
        dist_func = lambda p, q: np.linalg.norm(p - q)

    # 1) 计算 MED 距离矩阵
    MED = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            MED[i, j] = dist_func(P[i], Q[j])

    # 2) 初始化动态规划矩阵
    D = np.full((m, n), np.inf)           # ACDTW 累积距离
    TP = np.zeros((m, n), dtype=int)      # 测试帧使用计数
    TQ = np.zeros((m, n), dtype=int)      # 模板帧使用计数
    trace = np.full((m, n), -1, dtype=int)  # 记录回溯方向: 0=diag, 1=up, 2=left

    # 起点
    D[0, 0] = MED[0, 0]
    TP[0, 0] = 1
    TQ[0, 0] = 1

    # 边界条件（第一行/第一列只能从对角/单方向扩展）
    for i in range(1, m):
        if window is not None and abs(i - 0) > window:
            continue
        # 从 (i-1, 0) 来 -> 测试帧 i 是重复使用
        N_prev = TP[i-1, 0] + TQ[i-1, 0]
        C = 2.0 * max(m, n) / (m + n) * N_prev
        D[i, 0] = MED[i, 0] + C * MED[i, 0] + D[i-1, 0]
        TP[i, 0] = TP[i-1, 0] + 1
        TQ[i, 0] = TQ[i-1, 0]      # 注意这里论文中 Q 不增加
        trace[i, 0] = 1            # 从上方来

    for j in range(1, n):
        if window is not None and abs(0 - j) > window:
            continue
        N_prev = TP[0, j-1] + TQ[0, j-1]
        C = 2.0 * max(m, n) / (m + n) * N_prev
        D[0, j] = MED[0, j] + C * MED[0, j] + D[0, j-1]
        TP[0, j] = TP[0, j-1]
        TQ[0, j] = TQ[0, j-1] + 1
        trace[0, j] = 2            # 从左边来

    # 3) 动态规划递推
    for i in range(1, m):
        for j in range(1, n):
            if window is not None and abs(i - j) > window:
                continue

            # 对角线移动 (i-1, j-1) -> (i, j)：无惩罚
            diag_cost = D[i-1, j-1]

            # 垂直移动 (i-1, j) -> (i, j)：测试帧 i 重复
            N_up = TP[i-1, j] + TQ[i-1, j]
            C_up = 2.0 * max(m, n) / (m + n) * N_up
            up_cost = D[i-1, j] + C_up * MED[i, j]

            # 水平移动 (i, j-1) -> (i, j)：模板帧 j 重复
            N_left = TP[i, j-1] + TQ[i, j-1]
            C_left = 2.0 * max(m, n) / (m + n) * N_left
            left_cost = D[i, j-1] + C_left * MED[i, j]

            costs = np.array([diag_cost, up_cost, left_cost])
            idx = np.argmin(costs)

            if idx == 0:
                D[i, j] = MED[i, j] + diag_cost
                TP[i, j] = 1
                TQ[i, j] = 1
                trace[i, j] = 0
            elif idx == 1:
                D[i, j] = MED[i, j] + up_cost
                TP[i, j] = TP[i-1, j] + 1
                TQ[i, j] = TQ[i-1, j]
                trace[i, j] = 1
            else:
                D[i, j] = MED[i, j] + left_cost
                TP[i, j] = TP[i, j-1]
                TQ[i, j] = TQ[i, j-1] + 1
                trace[i, j] = 2

    # 4) 回溯最优路径
    best_path = []
    i, j = m - 1, n - 1
    best_path.append((i, j))

    while i > 0 or j > 0:
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        else:
            j -= 1
        best_path.append((i, j))

    best_path.reverse()
    return np.array(best_path), MED, TP, TQ
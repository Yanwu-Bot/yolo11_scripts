import numpy as np

def acdtw(P, Q, dist_func=None, window=None, penalty_coef=0.8):
    m, n = len(P), len(Q)
    if window is not None:
        window = max(window, abs(m - n))

    if dist_func is None:
        dist_func = lambda p, q: np.linalg.norm(p - q)

    MED = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            MED[i, j] = dist_func(P[i], Q[j])

    D = np.full((m, n), np.inf)
    TP = np.zeros((m, n), dtype=int)
    TQ = np.zeros((m, n), dtype=int)
    trace = np.full((m, n), -1, dtype=int)

    D[0, 0] = MED[0, 0]
    TP[0, 0] = 1
    TQ[0, 0] = 1

    coef = penalty_coef * 2.0 * max(m, n) / (m + n)

    # 第一列 j=0：从上方来 = 模板帧0被复用 → TQ累加，TP重置
    for i in range(1, m):
        if window is not None and i > window:
            continue
        N_prev = TP[i-1, 0] + TQ[i-1, 0]
        C = coef * N_prev
        D[i, 0] = MED[i, 0] + C * MED[i, 0] + D[i-1, 0]
        TP[i, 0] = 1                  # 测试帧 i 新来，重置
        TQ[i, 0] = TQ[i-1, 0] + 1     # 模板帧 0 被复用，累加
        trace[i, 0] = 1

    # 第一行 i=0：从左边来 = 测试帧0被复用 → TP累加，TQ重置
    for j in range(1, n):
        if window is not None and j > window:
            continue
        N_prev = TP[0, j-1] + TQ[0, j-1]
        C = coef * N_prev
        D[0, j] = MED[0, j] + C * MED[0, j] + D[0, j-1]
        TP[0, j] = TP[0, j-1] + 1     # 测试帧 0 被复用，累加
        TQ[0, j] = 1                  # 模板帧 j 新来，重置
        trace[0, j] = 2

    # 主循环
    for i in range(1, m):
        for j in range(1, n):
            if window is not None and abs(i - j) > window:
                continue

            diag_cost = D[i-1, j-1]

            N_up = TP[i-1, j] + TQ[i-1, j]
            C_up = coef * N_up
            up_cost = D[i-1, j] + C_up * MED[i, j]

            N_left = TP[i, j-1] + TQ[i, j-1]
            C_left = coef * N_left
            left_cost = D[i, j-1] + C_left * MED[i, j]

            costs = np.array([diag_cost, up_cost, left_cost])
            idx = np.argmin(costs)

            if idx == 0:
                D[i, j] = MED[i, j] + diag_cost
                TP[i, j] = 1
                TQ[i, j] = 1
                trace[i, j] = 0
            elif idx == 1:  # 上方：模板帧 j 被复用
                D[i, j] = MED[i, j] + up_cost
                TP[i, j] = 1
                TQ[i, j] = TQ[i-1, j] + 1
                trace[i, j] = 1
            else:           # 左边：测试帧 i 被复用
                D[i, j] = MED[i, j] + left_cost
                TP[i, j] = TP[i, j-1] + 1
                TQ[i, j] = 1
                trace[i, j] = 2

    # 回溯
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
import sys
from itertools import product

import numpy as np
import pandas as pd

import DINA_RanGen


# def calculate_savg(A,Q):
#     n_stu = A.shape[1]
#     n_que = Q.shape[1]
#     n_kno = A.shape[0]
#     savg =np.zeros((n_stu,n_que))
#
#     for u in n_stu:
#         for v in n_que:
#             # 对Q第v行求和
#             row_index = v  # 行索引从0开始
#             row_k_sum = np.sum(Q[row_index, :])
#             Inuv_total=1.0
#             for k in n_kno:
#                 Inuv_total*=A[u][k]**Q[v][k]
#             savg[u][v]=Inuv_total**(1/row_k_sum)
#
#     return savg
#
#
# def calculate_A_real(savg,s,g,X):
#     n_stu = X.shape[0]
#     n_que = X.shape[1]
#     A_real= np.zeros((n_stu,n_que))
#
#     for u in n_stu:
#         for v in n_que:
#             if X[u][v]==1:
#                 distributer = ((1 - s[v]) * savg[u][v] + g[v]*(1- savg[u][v]))
#                 if distributer!=0 :
#                     A_real[u][v] = (1 - s[v]) * savg[u][v]/distributer
#                 else:
#                     print("distributer 0!!!!!!")
#
#             if X[u][v]==0:
#                 distributer = (s[v] * savg[u][v] + (1-g[v])*(1- savg[u][v]))
#                 if distributer!=0 :
#                     A_real[u][v] =s [v] * savg[u][v]
#                 else:
#                     print("distributer 0!!!!!!")
#
#     return A_real
#
# def CD(X,Q):
#
#     A_all = np.array(list(product([0, 1], repeat=Q.shape[0])))
#     priors = DINA_RanGen.get_priors(A_all, p_know=0.7, p_know_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
#
#     pi_t, g_t, s_t, gamma = DINA_RanGen.em(X,Q, maxIter=10000, tol=1e-6, prior=priors[2])
#     s = []
#     g = []
#     s.append(s_t)
#     g.append(g_t)
#     A, A_idx = DINA_RanGen.solve(gamma, Q.shape[0])
#     A_real = calculate_A_real(calculate_savg(A, Q), s, g, X)
#
#     return A_real


class CD:
    def __init__(self, X, Q):
        self.X = X
        self.Q = Q
        self.A_real = None  # 初始化为 None，稍后计算
        self.n_stu = X.shape[0]
        self.n_que = Q.shape[1]
        self.n_kno = Q.shape[0]


    def calculate_savg(self, A, Q):


        savg = np.zeros((self.n_stu, self.n_que))

        for u in range(self.n_stu):
            for v in range(self.n_que):
                row_index = v
                row_k_sum = np.sum(Q[:, row_index])
                Inuv_total = 1.0
                for k in range(self.n_kno):  # 修正：应使用 range
                    Inuv_total *= A[u][k] ** Q[k][v]
                if row_k_sum != 0:
                    savg[u][v] = Inuv_total ** (1 / row_k_sum)
                else:
                    print("this question is invalid!")
                    sys.exit()
        return savg

    def calculate_A_real(self, savg, s, g, X):

        A_real = np.zeros((self.n_stu, self.n_que))

        for u in range(self.n_stu):
            for v in range(self.n_que):
                print(f"s: {s}, g: {g}, savg: {savg[u][v]}, u: {u}, v: {v}")

                if X[u][v] == 1:
                    distributer = ((1 - s[v]) * savg[u][v] + g[v] * (1 - savg[u][v]))
                    print("distributer:")
                    print(distributer)
                    if distributer != 0:
                        A_real[u][v] = (1 - s[v]) * savg[u][v] / distributer
                    else:
                        print("distributer 0!!!!!!")

                if X[u][v] == 0:
                    distributer = (s[v] * savg[u][v] + (1 - g[v]) * (1 - savg[u][v]))
                    print("distributer:")
                    print(distributer)
                    if distributer != 0:
                        A_real[u][v] = s[v] * savg[u][v]
                    else:
                        print("distributer 0!!!!!!")

        return A_real

    def calculate_A_real_values(self):  # 新增方法
        A_all = np.array(list(product([0, 1], repeat=self.Q.shape[0])))
        priors = DINA_RanGen.get_priors(A_all, p_know=0.7, p_know_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        pi_t, g_t, s_t, gamma = DINA_RanGen.em(self.X, self.Q, maxIter=10000, tol=1e-6, prior=priors[2])
        s = []
        g = []
        # s.append(s_t)
        # g.append(g_t)
        s = s_t
        g = g_t
        A, A_idx = DINA_RanGen.solve(gamma, self.Q.shape[0])
        savg=self.calculate_savg(A, self.Q)
        self.A_real = self.calculate_A_real(savg, s, g, self.X)

        return self.A_real











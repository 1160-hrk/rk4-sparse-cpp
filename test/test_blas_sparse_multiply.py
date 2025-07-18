import numpy as np
import scipy.sparse as sp
from rk4_sparse._rk4_sparse_cpp import test_blas_sparse_multiply

# 3x3行列
# [[1+2j, 0,   3+4j],
#  [0,    5+6j, 0   ],
#  [0,    0,    0   ]]
data = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
indices = np.array([0, 2, 1], dtype=np.int32)
indptr = np.array([0, 2, 3, 3], dtype=np.int32)
H0 = sp.csr_matrix((data, indices, indptr), shape=(3, 3))

# テスト用ベクトル
psi0 = np.array([1+0j, 0+1j, 1-1j], dtype=np.complex128)

# BLAS最適化版のスパース行列-ベクトル積を実行
result = test_blas_sparse_multiply(H0, psi0)

# 期待値（scipyで計算 + 虚数単位を掛ける）
expected = -1j * H0.dot(psi0)

print("H0.toarray():\n", H0.toarray())
print("psi0:", psi0)
print("result:", result)
print("expected:", expected)

# 検証
assert np.allclose(result, expected), "BLASスパース行列-ベクトル積の結果が一致しません" 
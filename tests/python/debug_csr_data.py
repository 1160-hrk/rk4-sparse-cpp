import numpy as np
import scipy.sparse as sp

# 3x3行列
# [[1+2j, 0,   3+4j],
#  [0,    5+6j, 0   ],
#  [0,    0,    0   ]]
data = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
indices = np.array([0, 2, 1], dtype=np.int32)
indptr = np.array([0, 2, 3, 3], dtype=np.int32)
H0 = sp.csr_matrix((data, indices, indptr), shape=(3, 3))

print("Python側のCSRデータ:")
print(f"rows={H0.shape[0]}, cols={H0.shape[1]}, nnz={H0.nnz}")
print(f"indptr: {H0.indptr}")
print(f"indices: {H0.indices}")
print(f"data: {H0.data}")

print("\n行列の内容:")
print(H0.toarray())

print("\n期待される計算結果:")
psi0 = np.array([1+0j, 0+1j, 1-1j], dtype=np.complex128)
expected = H0.dot(psi0)
print(f"psi0: {psi0}")
print(f"expected: {expected}")

# 手動計算で確認
print("\n手動計算:")
result = np.zeros(3, dtype=np.complex128)
for i in range(3):
    start = H0.indptr[i]
    end = H0.indptr[i + 1]
    print(f"行{i}: indptr[{i}]={start}, indptr[{i+1}]={end}")
    for j in range(start, end):
        col_idx = H0.indices[j]
        val = H0.data[j]
        print(f"  列{col_idx}: {val} * {psi0[col_idx]} = {val * psi0[col_idx]}")
        result[i] += val * psi0[col_idx]
print(f"手動計算結果: {result}")

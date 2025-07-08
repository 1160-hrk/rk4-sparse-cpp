import numpy as np
from scipy.sparse import csc_matrix
from python import rk4_propagate

dim = 4
H0 = csc_matrix(np.eye(dim, dtype=np.complex128))
mux = csc_matrix(np.eye(dim, dtype=np.complex128))
muy = csc_matrix(np.eye(dim, dtype=np.complex128))

print("H0 type:", type(H0))
print("H0 dtype:", H0.dtype)
print("H0 shape:", H0.shape)
print("H0 nnz:", H0.nnz)

print("mux type:", type(mux))
print("mux dtype:", mux.dtype)
print("mux shape:", mux.shape)
print("mux nnz:", mux.nnz)

psi0 = np.ones(dim, dtype=np.complex128)
print("psi0 type:", type(psi0))
print("psi0 dtype:", psi0.dtype)
print("psi0 shape:", psi0.shape)

# 2次元配列に変換
psi0 = psi0.reshape(-1, 1)
print("reshaped psi0 type:", type(psi0))
print("reshaped psi0 dtype:", psi0.dtype)
print("reshaped psi0 shape:", psi0.shape)

Ex3 = [[1.0, 1.0, 1.0]] * 10
Ey3 = [[0.0, 0.0, 0.0]] * 10

print("Ex3 type:", type(Ex3))
print("Ex3 length:", len(Ex3))
print("Ex3[0] type:", type(Ex3[0]))
print("Ex3[0]:", Ex3[0])

try:
out = rk4_propagate(
    H0, mux, muy, psi0,
    Ex3, Ey3,
    0.01, 10, 1,
    False, True
)
    print("Output shape:", out.shape)
print(out)
except Exception as e:
    print("Error:", e)

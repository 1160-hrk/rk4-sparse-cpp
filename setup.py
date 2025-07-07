"""パッケージのセットアップスクリプト"""

from setuptools import setup, Extension
import numpy as np

# C++拡張モジュールの設定
ext_modules = [
    Extension(
        name="python._excitation_rk4_sparse",  # pythonパッケージ直下に配置
        sources=[
            "src/bindings.cpp",
            "src/excitation_rk4_sparse.cpp"
        ],
        include_dirs=[
            "include",
            np.get_include(),
            "/usr/include/eigen3"  # Eigenのインクルードパスを追加
        ],
        extra_compile_args=["-std=c++17", "-O3", "-march=native"],
        language="c++"
    )
]

setup(
    name="excitation-rk4-sparse",
    version="0.1.0",
    packages=["python"],  # pythonパッケージのみを指定
    ext_modules=ext_modules,
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0"
    ],
    python_requires=">=3.8"
)

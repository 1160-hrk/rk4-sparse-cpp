#!/usr/bin/env python3
"""
Trajectory Comparison - julia_killer_rk4.cppとexcitation_rk4_sparse.cppの実装のトラジェクトリー比較
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(savepath, exist_ok=True)

import numpy as np
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# C++実装のインポート
erk4 = None
try:
    import rk4_sparse._rk4_sparse_cpp as erk4
    print("✅ C++実装が利用可能です")
    if erk4:
        available_functions = [func for func in dir(erk4) if func.startswith('rk4_') or func.startswith('julia_')]
        print(f"利用可能な関数: {available_functions}")
except ImportError as e:
    print(f"❌ C++実装が見つかりません: {e}")

# Python実装のインポート
rk4_sparse_py = None
rk4_numba_py = None
try:
    from rk4_sparse import rk4_sparse_py, rk4_numba_py
    print("✅ Python実装が利用可能です")
except ImportError as e:
    print(f"❌ Python実装が見つかりません: {e}")

if erk4 is None and rk4_sparse_py is None:
    print("❌ 利用可能な実装がありません")
    sys.exit(1)


def create_test_system():
    """テスト用の二準位系を作成"""
    # シミュレーションパラメータ
    omega = 1.0  # 遷移周波数
    E0 = 0.1     # 電場強度
    omega_L = 1.0  # レーザー周波数（共鳴条件）
    dt_E = 0.01     # 時間ステップ
    steps_E = 1000  # 総ステップ数（比較用に短縮）
    stride = 1    # 出力間隔

    # 二準位系のハミルトニアン
    H0 = csr_matrix([[0, 0],
                     [0, omega]], dtype=np.complex128)

    # 双極子演算子（x方向）
    mux = csr_matrix([[0, 1],
                      [1, 0]], dtype=np.complex128)

    # 双極子演算子（y方向）- この例では使用しない
    muy = csr_matrix([[0, 0],
                      [0, 0]], dtype=np.complex128)

    # 初期状態 (基底状態)
    psi0 = np.array([1, 0], dtype=np.complex128)

    # 正弦波の電場を生成
    t = np.arange(0, dt_E * (steps_E+2), dt_E)
    Ex = E0 * np.sin(omega_L * t) 
    Ey = np.zeros_like(Ex)

    return H0, mux, muy, Ex, Ey, psi0, dt_E, steps_E, stride, t, E0, omega_L


def run_implementations(H0, mux, muy, Ex, Ey, psi0, dt_E, steps_E, stride, E0, omega_L):
    """各実装を実行して結果を比較"""
    results = {}
    times = {}
    
    # 実装リスト（利用可能なもののみ）
    implementations = []
    
    # Python実装
    if rk4_numba_py is not None:
        implementations.append(("Python-Numba", "rk4_numba_py"))
    if rk4_sparse_py is not None:
        implementations.append(("Python-Sparse", "rk4_sparse_py"))
    
    # C++実装（利用可能なもののみ）
    if erk4 is not None:
        if hasattr(erk4, 'rk4_sparse_eigen'):
            implementations.append(("C++-Eigen", "rk4_sparse_eigen"))
        if hasattr(erk4, 'rk4_sparse_eigen_cached'):
            implementations.append(("C++-Eigen-Cached", "rk4_sparse_eigen_cached"))
        # Julia Killer Phase 1（高次元で実行、2次元を抽出）
        if hasattr(erk4, 'julia_killer_rk4_phase1'):
            implementations.append(("Julia-Killer-Phase1", "julia_killer_rk4_phase1"))
    
    print(f"\n実行する実装: {[name for name, _ in implementations]}")
    
    for name, func_name in implementations:
        print(f"\n実行中: {name}")
        try:
            start_t = time.perf_counter()
            if func_name == "rk4_numba_py" and rk4_numba_py is not None:
                H0_numba = H0.toarray()
                mux_numba = mux.toarray()
                muy_numba = muy.toarray()
                result = rk4_numba_py(
                    H0_numba, mux_numba, muy_numba,
                    Ex.astype(np.float64), Ey.astype(np.float64),
                    psi0,
                    dt_E*2,
                    True,
                    stride,
                    False,
                )
            elif func_name == "rk4_sparse_py" and rk4_sparse_py is not None:
                result = rk4_sparse_py(
                    H0, mux, muy,
                    Ex, Ey,
                    psi0,
                    dt_E*2,
                    True,
                    stride,
                    False,
                )
            elif func_name == "julia_killer_rk4_phase1":
                # julia_killer_rk4_phase1は高次元スパース行列専用
                # 16次元で実行し、結果の最初の2次元を抽出
                julia_steps = 501  # julia_killer_test.pyと同じ
                julia_dt = 0.01    # julia_killer_test.pyと同じ
                
                # 16次元のスパース行列を生成（julia_killer_test.pyと同じ構造）
                from scipy.sparse import csr_matrix
                
                # 16次元のハミルトニアン: 対角成分 E_n = n * hbar * omega
                dim = 16
                H0_data = []
                H0_row = []
                H0_col = []
                
                for i in range(dim):
                    H0_data.append(complex(i * 1.0, 0))  # エネルギー準位
                    H0_row.append(i)
                    H0_col.append(i)
                
                # 双極子行列: 最近接遷移のみ
                mux_data = []
                mux_row = []
                mux_col = []
                
                muy_data = []
                muy_row = []
                muy_col = []
                
                for i in range(dim - 1):
                    # X方向双極子モーメント
                    mux_data.extend([complex(np.sqrt(i+1), 0), complex(np.sqrt(i+1), 0)])
                    mux_row.extend([i, i+1])
                    mux_col.extend([i+1, i])
                    
                    # Y方向双極子モーメント  
                    muy_data.extend([complex(0, np.sqrt(i+1)), complex(0, -np.sqrt(i+1))])
                    muy_row.extend([i, i+1])
                    muy_col.extend([i+1, i])
                
                # CSR形式のスパース行列として生成
                H0_julia = csr_matrix((H0_data, (H0_row, H0_col)), shape=(dim, dim), dtype=complex)
                mux_julia = csr_matrix((mux_data, (mux_row, mux_col)), shape=(dim, dim), dtype=complex)
                muy_julia = csr_matrix((muy_data, (muy_row, muy_col)), shape=(dim, dim), dtype=complex)
                
                # julia_killer_test.pyと同じ電場生成方法
                t_julia = np.linspace(0, 10, julia_steps)  # julia_killer_test.pyと同じ
                Ex_julia = E0 * np.sin(omega_L * t_julia)
                Ey_julia = np.zeros_like(Ex_julia)
                
                # 初期状態（基底状態）
                psi0_julia = np.zeros(dim, dtype=complex)
                psi0_julia[0] = 1.0
                
                print(f"  Julia Killer用電場サイズ: {Ex_julia.size} (期待値: {julia_steps})")
                print(f"  Julia Killer用行列次元: {dim}")
                
                result = erk4.julia_killer_rk4_phase1(
                    H0_julia, mux_julia, muy_julia,
                    Ex_julia, Ey_julia,
                    psi0_julia,
                    julia_dt,
                    True,   # return_traj
                    stride,
                    False   # renorm
                )
                
                # 16次元の結果から最初の2次元を抽出
                if result.shape[0] == 250 and result.shape[1] == 16:
                    # 最初の2次元のみを抽出
                    result_2d = result[:, :2]
                    
                    # 250ステップを501ステップに補間
                    from scipy.interpolate import interp1d
                    t_250 = np.linspace(0, 10, 250)
                    t_501 = np.linspace(0, 10, 501)
                    
                    result_interp = np.zeros((501, 2), dtype=complex)
                    for i in range(2):
                        f_interp = interp1d(t_250, result_2d[:, i], kind='linear')
                        result_interp[:, i] = f_interp(t_501)
                    
                    result = result_interp
                    print(f"  結果を補間: {result.shape}")
            else:
                # C++実装
                func = getattr(erk4, func_name)
                result = func(
                    H0, mux, muy,
                    Ex, Ey,
                    psi0,
                    dt_E*2,
                    True,
                    stride,
                    False,
                )
            end_t = time.perf_counter()
            results[name] = result
            times[name] = end_t - start_t
            print(f"✅ {name} 完了: {times[name]:.3f}秒, 結果形状: {result.shape}")
        except Exception as e:
            print(f"❌ {name} エラー: {e}")
            results[name] = None
            times[name] = float('inf')
            continue
    return results, times


def analyze_trajectories(results, times, t, dt_E, steps_E, stride):
    """トラジェクトリーの解析と比較"""
    print("\n=== トラジェクトリー解析 ===")
    
    # 成功した実装のみを抽出
    successful_impls = {name: result for name, result in results.items() if result is not None}
    
    if len(successful_impls) < 2:
        print("❌ 比較可能な実装が2つ未満です")
        return None
    
    print(f"比較対象実装: {list(successful_impls.keys())}")
    
    # 基底状態と励起状態の占有数を計算
    populations = {}
    for name, result in successful_impls.items():
        if result is not None:
            P0 = np.abs(result[:, 0])**2  # 基底状態
            P1 = np.abs(result[:, 1])**2  # 励起状態
            populations[name] = {'ground': P0, 'excited': P1}
    
    # 時間軸の調整
    t_analytical = np.arange(0, dt_E * (steps_E+1), 2*dt_E*stride)
    
    # ラビ振動の解析解
    E0 = 0.1
    Omega_R = E0 / 2  # 実効的なラビ周波数
    P0_analytical = np.cos(Omega_R * t_analytical)**2
    P1_analytical = np.sin(Omega_R * t_analytical)**2
    
    # 実装間の差を計算
    impl_names = list(successful_impls.keys())
    max_differences = {}
    
    for i, name1 in enumerate(impl_names):
        for j, name2 in enumerate(impl_names):
            if i < j:  # 重複を避ける
                diff_ground = np.max(np.abs(populations[name1]['ground'] - populations[name2]['ground']))
                diff_excited = np.max(np.abs(populations[name1]['excited'] - populations[name2]['excited']))
                max_differences[f"{name1}_vs_{name2}"] = {
                    'ground': diff_ground,
                    'excited': diff_excited
                }
    
    # 解析解との差を計算
    analytical_differences = {}
    for name in impl_names:
        diff_ground = np.max(np.abs(populations[name]['ground'] - P0_analytical))
        diff_excited = np.max(np.abs(populations[name]['excited'] - P1_analytical))
        analytical_differences[name] = {
            'ground': diff_ground,
            'excited': diff_excited
        }
    
    # 結果の表示
    print("\n=== 実行時間 ===")
    for name, time_val in times.items():
        if time_val != float('inf'):
            print(f"{name}: {time_val:.3f}秒")
    
    print("\n=== 実装間の最大差 ===")
    for pair, diffs in max_differences.items():
        print(f"{pair}:")
        print(f"  基底状態: {diffs['ground']:.2e}")
        print(f"  励起状態: {diffs['excited']:.2e}")
    
    print("\n=== 解析解との最大差 ===")
    for name, diffs in analytical_differences.items():
        print(f"{name}:")
        print(f"  基底状態: {diffs['ground']:.2e}")
        print(f"  励起状態: {diffs['excited']:.2e}")
    
    # 最終状態の比較
    print("\n=== 最終状態の比較 ===")
    for name in impl_names:
        final_ground = populations[name]['ground'][-1]
        final_excited = populations[name]['excited'][-1]
        print(f"{name}: 基底状態={final_ground:.6f}, 励起状態={final_excited:.6f}")
    
    print(f"解析解: 基底状態={P0_analytical[-1]:.6f}, 励起状態={P1_analytical[-1]:.6f}")
    
    return populations, max_differences, analytical_differences, t_analytical, P0_analytical, P1_analytical


def plot_comparison(populations, max_differences, analytical_differences, t_analytical, P0_analytical, P1_analytical, savepath):
    """比較結果のプロット"""
    impl_names = list(populations.keys())
    
    if len(impl_names) == 0:
        print("❌ プロット可能な実装がありません")
        return
    
    # プロットの設定
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trajectory Comparison: julia_killer_rk4 vs excitation_rk4_sparse', fontsize=16)
    
    # 色の設定
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = colors[:len(impl_names)]
    
    # サブプロット1: 基底状態の占有数
    ax1 = axes[0, 0]
    for i, name in enumerate(impl_names):
        ax1.plot(t_analytical, populations[name]['ground'], 
                color=colors[i], label=f'{name} (Ground)', alpha=0.8)
    ax1.plot(t_analytical, P0_analytical, 'k--', label='Analytical (Ground)', alpha=0.7)
    ax1.set_xlabel('Time (a.u.)')
    ax1.set_ylabel('Ground State Population')
    ax1.set_title('Ground State Population')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # サブプロット2: 励起状態の占有数
    ax2 = axes[0, 1]
    for i, name in enumerate(impl_names):
        ax2.plot(t_analytical, populations[name]['excited'], 
                color=colors[i], label=f'{name} (Excited)', alpha=0.8)
    ax2.plot(t_analytical, P1_analytical, 'k--', label='Analytical (Excited)', alpha=0.7)
    ax2.set_xlabel('Time (a.u.)')
    ax2.set_ylabel('Excited State Population')
    ax2.set_title('Excited State Population')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # サブプロット3: 実装間の差
    ax3 = axes[1, 0]
    if len(impl_names) >= 2:
        # 最初の2つの実装の差を表示
        name1, name2 = impl_names[0], impl_names[1]
        diff_ground = np.abs(populations[name1]['ground'] - populations[name2]['ground'])
        diff_excited = np.abs(populations[name1]['excited'] - populations[name2]['excited'])
        
        ax3.plot(t_analytical, diff_ground, 'b-', label=f'{name1} vs {name2} (Ground)', alpha=0.7)
        ax3.plot(t_analytical, diff_excited, 'r-', label=f'{name1} vs {name2} (Excited)', alpha=0.7)
        ax3.set_xlabel('Time (a.u.)')
        ax3.set_ylabel('|Population Difference|')
        ax3.set_title('Implementation Differences')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_yscale('log')
    
    # サブプロット4: 解析解との差
    ax4 = axes[1, 1]
    for i, name in enumerate(impl_names):
        diff_ground = np.abs(populations[name]['ground'] - P0_analytical)
        diff_excited = np.abs(populations[name]['excited'] - P1_analytical)
        
        ax4.plot(t_analytical, diff_ground, color=colors[i], 
                label=f'{name} vs Analytical (Ground)', alpha=0.7)
        ax4.plot(t_analytical, diff_excited, color=colors[i], linestyle='--',
                label=f'{name} vs Analytical (Excited)', alpha=0.7)
    
    ax4.set_xlabel('Time (a.u.)')
    ax4.set_ylabel('|Population Difference|')
    ax4.set_title('Difference from Analytical Solution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'trajectory_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nプロットを保存しました: {os.path.join(savepath, 'trajectory_comparison.png')}")


def main():
    """メイン関数"""
    print("🚀 Trajectory Comparison: julia_killer_rk4 vs excitation_rk4_sparse")
    print("=" * 80)
    
    # テストシステムの作成
    print("\n📊 テストシステムを作成中...")
    H0, mux, muy, Ex, Ey, psi0, dt_E, steps_E, stride, t, E0, omega_L = create_test_system()
    
    print(f"システム情報:")
    print(f"  次元数: {H0.shape[0]}")
    print(f"  時間ステップ数: {steps_E}")
    print(f"  時間刻み: {dt_E}")
    print(f"  総時間: {t[-1]:.3f}")
    print(f"  電場強度: {np.max(Ex):.3f}")
    
    # 各実装の実行
    print("\n⚡ 各実装を実行中...")
    results, times = run_implementations(H0, mux, muy, Ex, Ey, psi0, dt_E, steps_E, stride, E0, omega_L)
    
    # トラジェクトリーの解析
    print("\n📈 トラジェクトリーを解析中...")
    analysis_result = analyze_trajectories(results, times, t, dt_E, steps_E, stride)
    
    if analysis_result is None:
        print("❌ 解析に失敗しました")
        return
    
    populations, max_differences, analytical_differences, t_analytical, P0_analytical, P1_analytical = analysis_result
    
    # プロットの作成
    print("\n📊 プロットを作成中...")
    plot_comparison(populations, max_differences, analytical_differences, 
                   t_analytical, P0_analytical, P1_analytical, savepath)
    
    # 結論の表示
    print("\n🎯 結論:")
    successful_impls = [name for name, result in results.items() if result is not None]
    
    if len(successful_impls) >= 2:
        # 最大差をチェック
        max_diff_threshold = 1e-10
        significant_differences = []
        
        for pair, diffs in max_differences.items():
            if diffs['ground'] > max_diff_threshold or diffs['excited'] > max_diff_threshold:
                significant_differences.append((pair, diffs))
        
        if significant_differences:
            print("⚠️  実装間に有意な差が見つかりました:")
            for pair, diffs in significant_differences:
                print(f"   {pair}: 基底状態差={diffs['ground']:.2e}, 励起状態差={diffs['excited']:.2e}")
        else:
            print("✅ すべての実装で一致した結果が得られました（数値誤差の範囲内）")
        
        # 最速実装の特定
        valid_times = {name: time_val for name, time_val in times.items() if time_val != float('inf')}
        if valid_times:
            fastest = min(valid_times.items(), key=lambda x: x[1])[0]
            print(f"🏆 最速実装: {fastest} ({valid_times[fastest]:.3f}秒)")
    
    print(f"\n📁 結果ファイル: {os.path.join(savepath, 'trajectory_comparison.png')}")


if __name__ == "__main__":
    main() 
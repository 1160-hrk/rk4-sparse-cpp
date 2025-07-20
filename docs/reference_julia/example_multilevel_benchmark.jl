using LinearAlgebra
using Pkg
Pkg.add("BenchmarkTools")
using BenchmarkTools
using SparseArrays
using Plots

include("rk_schrodinger.jl") 
include("plots_config.jl")
using .PlotsConfig

# -------------------------------------------
# 多準位系の定義（最近接遷移のみ許容）
# -------------------------------------------

function create_multilevel_system(dim::Int, hbar::Float64=1.0, omega::Float64=1.0, mu::Float64=0.1)
    """
    多準位系のハミルトニアンと双極子行列を生成
    - 対角成分: E_n = n * hbar * omega
    - 双極子行列: 最近接準位間のみ遷移可能
    """
    
    # ハミルトニアン（対角成分）
    H0 = Diagonal(ComplexF64[i * hbar * omega for i in 0:dim-1])
    
    # 双極子行列（最近接遷移のみ）
    mux = zeros(ComplexF64, dim, dim)
    for i in 1:dim-1
        mux[i, i+1] = mu * sqrt(i)  # 上向き遷移
        mux[i+1, i] = mu * sqrt(i)  # 下向き遷移
    end
    
    muy = zeros(ComplexF64, dim, dim)  # y 成分なし
    
    return H0, mux, muy
end

# -------------------------------------------
# 電場設定
# -------------------------------------------

T = 10.0  # 全体時間
dt = 0.01
t = 0:dt:T
steps = length(t)

E0 = 1.0  # 電場振幅
omega_L = 1.0  # 電場周波数

Ex = E0 .* sin.(omega_L .* t)
Ey = zeros(length(t))  # y 成分なし

# -------------------------------------------
# 初期状態
# -------------------------------------------

function create_initial_state(dim::Int)
    """基底状態を初期状態とする"""
    psi0 = zeros(ComplexF64, dim)
    psi0[1] = 1.0
    return psi0
end

# -------------------------------------------
# ベンチマーク関数
# -------------------------------------------

function run_benchmark(dim::Int, method::Symbol)
    """指定された次元と手法でベンチマークを実行"""
    
    # システムの生成
    H0, mux, muy = create_multilevel_system(dim)
    psi0 = create_initial_state(dim)
    
    # 行列をsparse形式に変換
    H0_sparse = sparse(H0)
    mux_sparse = sparse(mux)
    muy_sparse = sparse(muy)
    
    stride = 1
    
    # 手法に応じてベンチマーク実行
    if method == :dense
        H0_dense = Matrix(H0)
        mux_dense = Matrix(mux)
        muy_dense = Matrix(muy)
        
        bench = @benchmark rk4_cpu(
            $H0_dense, $mux_dense, $muy_dense,
            $Ex, $Ey, $psi0, $dt*2;
            return_traj=true, stride=$stride, renorm=false
        )
        
    elseif method == :sparse
        bench = @benchmark rk4_cpu_sparse(
            $H0_sparse, $mux_sparse, $muy_sparse,
            $Ex, $Ey, $psi0, $dt*2;
            return_traj=true, stride=$stride, renorm=false
        )
        
    elseif method == :multithread
        bench = @benchmark rk4_cpu_sparse_mt(
            $H0_sparse, $mux_sparse, $muy_sparse,
            $Ex, $Ey, $psi0, $dt*2;
            return_traj=true, stride=$stride, renorm=false
        )
        
    elseif method == :simd
        bench = @benchmark rk4_cpu_sparse_simd(
            $H0_sparse, $mux_sparse, $muy_sparse,
            $Ex, $Ey, $psi0, $dt*2;
            return_traj=true, stride=$stride, renorm=false
        )
    end
    
    return bench
end

# -------------------------------------------
# ベンチマーク実行
# -------------------------------------------

println("多準位系ベンチマーク開始")
println("="^50)

# テストする次元
dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
methods = [:dense, :sparse, :multithread, :simd]

# 手法名とラベルの対応
method_labels = Dict(
    :dense => "Dense Matrix",
    :sparse => "Sparse Matrix", 
    :multithread => "Multithread",
    :simd => "SIMD Optimized"
)

# 結果を格納する配列
results = Dict()

for dim in dimensions
    println("\n次元: $dim")
    println("-"^30)
    
    results[dim] = Dict()
    
    for method in methods
        println("手法: $method")
        
        try
            bench = run_benchmark(dim, method)
            
            results[dim][method] = Dict(
                "mean_time" => mean(bench.times) / 1e6,  # ms
                "min_time" => minimum(bench.times) / 1e6,
                "max_time" => maximum(bench.times) / 1e6,
                "std_time" => std(bench.times) / 1e6,
                "memory" => bench.memory / 1024,  # KiB
                "allocs" => bench.allocs
            )
            
            println("  平均時間: $(results[dim][method]["mean_time"]) ms")
            println("  メモリ使用量: $(results[dim][method]["memory"]) KiB")
            
        catch e
            println("  エラー: $e")
            results[dim][method] = nothing
        end
    end
end

# -------------------------------------------
# 結果の可視化
# -------------------------------------------

println("\n結果の可視化")
println("="^50)

# 実行時間の比較プロット
p1 = plot(
    xlabel="Dimension",
    ylabel="Mean Execution Time (ms)",
    title="Multilevel System: Execution Time Comparison",
    xscale=:log10,
    yscale=:log10,
    legend=:topleft
)

for method in methods
    times = []
    dims = []
    
    for dim in dimensions
        if results[dim][method] !== nothing
            push!(times, results[dim][method]["mean_time"])
            push!(dims, dim)
        end
    end
    
    if !isempty(times)
        plot!(p1, dims, times, 
              label=method_labels[method], 
              marker=:circle, 
              markersize=4,
              linewidth=2)
    end
end

display(p1)

# メモリ使用量の比較プロット
p2 = plot(
    xlabel="Dimension",
    ylabel="Memory Usage (KiB)",
    title="Multilevel System: Memory Usage Comparison",
    xscale=:log10,
    yscale=:log10,
    legend=:topleft
)

for method in methods
    memory = []
    dims = []
    
    for dim in dimensions
        if results[dim][method] !== nothing
            push!(memory, results[dim][method]["memory"])
            push!(dims, dim)
        end
    end
    
    if !isempty(memory)
        plot!(p2, dims, memory, 
              label=method_labels[method], 
              marker=:square, 
              markersize=4,
              linewidth=2)
    end
end

display(p2)

# スケーリング解析
println("\nスケーリング解析")
println("-"^30)

for method in methods
    println("\n手法: $(method_labels[method])")
    
    # 実行時間のスケーリング
    times = []
    dims = []
    
    for dim in dimensions
        if results[dim][method] !== nothing
            push!(times, results[dim][method]["mean_time"])
            push!(dims, dim)
        end
    end
    
    if length(times) >= 2
        # 対数スケールでの線形フィッティング
        log_times = log10.(times)
        log_dims = log10.(dims)
        
        # 簡単な線形回帰
        n = length(log_times)
        sum_x = sum(log_dims)
        sum_y = sum(log_times)
        sum_xy = sum(log_dims .* log_times)
        sum_x2 = sum(log_dims.^2)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
        intercept = (sum_y - slope * sum_x) / n
        
        println("  実行時間スケーリング: O(dim^$(round(slope, digits=2)))")
        println("  フィッティング式: time = $(round(10^intercept, digits=3)) × dim^$(round(slope, digits=2))")
    end
end

# 詳細な結果テーブル
println("\n詳細な結果")
println("="^50)

println("次元\t手法\t\t\t平均時間(ms)\t最小時間(ms)\t最大時間(ms)\t標準偏差(ms)\tメモリ(KiB)\tアロケーション回数")
println("-"^120)

for dim in dimensions
    for method in methods
        if results[dim][method] !== nothing
            r = results[dim][method]
            println("$dim\t$(method_labels[method])\t\t$(round(r["mean_time"], digits=3))\t\t$(round(r["min_time"], digits=3))\t\t$(round(r["max_time"], digits=3))\t\t$(round(r["std_time"], digits=3))\t\t$(round(r["memory"], digits=1))\t\t$(r["allocs"])")
        else
            println("$dim\t$(method_labels[method])\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\tN/A")
        end
    end
end

# システム情報
println("\nシステム情報")
println("="^50)
println("利用可能なスレッド数: ", Threads.nthreads())
println("BLASスレッド数: ", BLAS.get_num_threads())
println("SIMD命令セット: ", LoopVectorization.VectorizationBase.register_size())

# グラフを保存
savefig(p1, "examples/figures/multilevel_execution_time.png")
savefig(p2, "examples/figures/multilevel_memory_usage.png")

println("\nグラフを保存しました:")
println("- examples/figures/multilevel_execution_time.png")
println("- examples/figures/multilevel_memory_usage.png") 
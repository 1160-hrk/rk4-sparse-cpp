using LinearAlgebra
using SparseArrays
using Base.Threads
using LoopVectorization

function rk4_cpu(
    H0::Matrix{ComplexF64},
    mux::Matrix{ComplexF64},
    muy::Matrix{ComplexF64},
    Ex::Vector{Float64},
    Ey::Vector{Float64},
    psi0::Vector{ComplexF64},
    dt::Float64;
    return_traj::Bool = true,
    stride::Int = 1,
    renorm::Bool = false
)
    steps = (length(Ex) - 1) ÷ 2
    Ex3 = hcat(Ex[1:2:end-2], Ex[2:2:end-1], Ex[3:2:end])
    Ey3 = hcat(Ey[1:2:end-2], Ey[2:2:end-1], Ey[3:2:end])

    psi = copy(psi0)
    dim = length(psi)
    n_out = steps ÷ stride + 1
    out = Matrix{ComplexF64}(undef, dim, n_out)
    out[:, 1] .= psi
    idx = 2

    buf = similar(psi)
    k1 = similar(psi)
    k2 = similar(psi)
    k3 = similar(psi)
    k4 = similar(psi)

    for s in 1:steps
        ex1, ex2, ex4 = Ex3[s, :]
        ey1, ey2, ey4 = Ey3[s, :]

        H1 = H0 .+ mux .* ex1 .+ muy .* ey1
        H2 = H0 .+ mux .* ex2 .+ muy .* ey2
        H4 = H0 .+ mux .* ex4 .+ muy .* ey4

        k1 .= -im .* (H1 * psi)
        buf .= psi .+ 0.5 * dt .* k1
        k2 .= -im .* (H2 * buf)
        buf .= psi .+ 0.5 * dt .* k2
        k3 .= -im .* (H2 * buf)
        buf .= psi .+ dt .* k3
        k4 .= -im .* (H4 * buf)

        psi .+= (dt / 6.0) .* (k1 .+ 2k2 .+ 2k3 .+ k4)

        if renorm
            psi ./= sqrt(real(psi' * psi))
        end

        if return_traj && (s % stride == 0)
            out[:, idx] .= psi
            idx += 1
        end
    end

    return return_traj ? out : reshape(psi, :, 1)
end


function rk4_cpu_sparse(
    H0::SparseMatrixCSC{ComplexF64,Int},
    mux::SparseMatrixCSC{ComplexF64,Int},
    muy::SparseMatrixCSC{ComplexF64,Int},
    Ex::Vector{Float64},
    Ey::Vector{Float64},
    psi0::Vector{ComplexF64},
    dt::Float64;
    return_traj::Bool = true,
    stride::Int = 1,
    renorm::Bool = false
)
    steps = (length(Ex) - 1) ÷ 2
    Ex3 = hcat(Ex[1:2:end-2], Ex[2:2:end-1], Ex[3:2:end])
    Ey3 = hcat(Ey[1:2:end-2], Ey[2:2:end-1], Ey[3:2:end])

    psi = copy(psi0)
    dim = length(psi)
    n_out = steps ÷ stride + 1
    out = Matrix{ComplexF64}(undef, dim, n_out)
    out[:, 1] .= psi
    idx = 2

    # バッファの初期化
    buf = similar(psi)
    k1 = similar(psi)
    k2 = similar(psi)
    k3 = similar(psi)
    k4 = similar(psi)

    # スパース行列の準備
    rows = Vector{Int}()
    cols = Vector{Int}()
    
    # 非ゼロ要素のパターンを収集
    for (i, j, v) in zip(findnz(H0)..., findnz(mux)..., findnz(muy)...)
        if !iszero(v)
            push!(rows, i)
            push!(cols, j)
        end
    end
    
    # 重複を除去
    unique_indices = unique(tuple.(rows, cols))
    rows = first.(unique_indices)
    cols = last.(unique_indices)
    
    # 各行列から値を抽出
    H0_data = [H0[i,j] for (i,j) in unique_indices]
    mux_data = [mux[i,j] for (i,j) in unique_indices]
    muy_data = [muy[i,j] for (i,j) in unique_indices]
    
    # 3つのスパース行列を事前に確保
    H1 = sparse(rows, cols, zeros(ComplexF64, length(rows)), dim, dim)
    H2 = sparse(rows, cols, zeros(ComplexF64, length(rows)), dim, dim)
    H4 = sparse(rows, cols, zeros(ComplexF64, length(rows)), dim, dim)
    
    # 値を格納するバッファ
    vals1 = Vector{ComplexF64}(undef, length(rows))
    vals2 = Vector{ComplexF64}(undef, length(rows))
    vals4 = Vector{ComplexF64}(undef, length(rows))

    for s in 1:steps
        ex1, ex2, ex4 = Ex3[s, :]
        ey1, ey2, ey4 = Ey3[s, :]

        # 行列要素の更新（nonzeros()を直接更新）
        @. vals1 = H0_data + mux_data * ex1 + muy_data * ey1
        @. vals2 = H0_data + mux_data * ex2 + muy_data * ey2
        @. vals4 = H0_data + mux_data * ex4 + muy_data * ey4
        
        nonzeros(H1) .= vals1
        nonzeros(H2) .= vals2
        nonzeros(H4) .= vals4

        k1 .= -im .* (H1 * psi)
        buf .= psi .+ 0.5 * dt .* k1

        k2 .= -im .* (H2 * buf)
        buf .= psi .+ 0.5 * dt .* k2

        k3 .= -im .* (H2 * buf)
        buf .= psi .+ dt .* k3

        k4 .= -im .* (H4 * buf)

        psi .+= (dt / 6.0) .* (k1 .+ 2k2 .+ 2k3 .+ k4)

        if renorm
            psi ./= sqrt(real(psi' * psi))
        end

        if return_traj && (s % stride == 0)
            out[:, idx] .= psi
            idx += 1
        end
    end

    return return_traj ? out : reshape(psi, :, 1)
end

function rk4_cpu_sparse_mt(
    H0::SparseMatrixCSC{ComplexF64,Int},
    mux::SparseMatrixCSC{ComplexF64,Int},
    muy::SparseMatrixCSC{ComplexF64,Int},
    Ex::Vector{Float64},
    Ey::Vector{Float64},
    psi0::Vector{ComplexF64},
    dt::Float64;
    return_traj::Bool = true,
    stride::Int = 1,
    renorm::Bool = false
)
    steps = (length(Ex) - 1) ÷ 2
    Ex3 = hcat(Ex[1:2:end-2], Ex[2:2:end-1], Ex[3:2:end])
    Ey3 = hcat(Ey[1:2:end-2], Ey[2:2:end-1], Ey[3:2:end])

    psi = copy(psi0)
    dim = length(psi)
    n_out = steps ÷ stride + 1
    out = Matrix{ComplexF64}(undef, dim, n_out)
    out[:, 1] .= psi
    idx = 2

    # バッファの初期化
    buf = similar(psi)
    k1 = similar(psi)
    k2 = similar(psi)
    k3 = similar(psi)
    k4 = similar(psi)

    # スパース行列の準備
    rows = Vector{Int}()
    cols = Vector{Int}()
    
    # 非ゼロ要素のパターンを収集
    for (i, j, v) in zip(findnz(H0)..., findnz(mux)..., findnz(muy)...)
        if !iszero(v)
            push!(rows, i)
            push!(cols, j)
        end
    end
    
    # 重複を除去
    unique_indices = unique(tuple.(rows, cols))
    rows = first.(unique_indices)
    cols = last.(unique_indices)
    
    # 各行列から値を抽出
    H0_data = [H0[i,j] for (i,j) in unique_indices]
    mux_data = [mux[i,j] for (i,j) in unique_indices]
    muy_data = [muy[i,j] for (i,j) in unique_indices]
    
    # 3つのスパース行列を事前に確保
    H1 = sparse(rows, cols, zeros(ComplexF64, length(rows)), dim, dim)
    H2 = sparse(rows, cols, zeros(ComplexF64, length(rows)), dim, dim)
    H4 = sparse(rows, cols, zeros(ComplexF64, length(rows)), dim, dim)
    
    # 値を格納するバッファ
    vals1 = Vector{ComplexF64}(undef, length(rows))
    vals2 = Vector{ComplexF64}(undef, length(rows))
    vals4 = Vector{ComplexF64}(undef, length(rows))

    # 行列要素の更新用バッファ
    chunk_size = length(rows) ÷ Threads.nthreads()
    
    for s in 1:steps
        ex1, ex2, ex4 = Ex3[s, :]
        ey1, ey2, ey4 = Ey3[s, :]

        # 行列要素の更新を並列化
        @threads for tid in 1:Threads.nthreads()
            start_idx = (tid - 1) * chunk_size + 1
            end_idx = tid == Threads.nthreads() ? length(rows) : tid * chunk_size
            
            @views begin
                @. vals1[start_idx:end_idx] = H0_data[start_idx:end_idx] + 
                    mux_data[start_idx:end_idx] * ex1 + 
                    muy_data[start_idx:end_idx] * ey1
                
                @. vals2[start_idx:end_idx] = H0_data[start_idx:end_idx] + 
                    mux_data[start_idx:end_idx] * ex2 + 
                    muy_data[start_idx:end_idx] * ey2
                
                @. vals4[start_idx:end_idx] = H0_data[start_idx:end_idx] + 
                    mux_data[start_idx:end_idx] * ex4 + 
                    muy_data[start_idx:end_idx] * ey4
            end
        end
        
        # スパース行列の更新
        nonzeros(H1) .= vals1
        nonzeros(H2) .= vals2
        nonzeros(H4) .= vals4

        # 行列-ベクトル積の計算（BLASが自動的に並列化）
        mul!(k1, H1, psi)
        k1 .*= -im
        
        @. buf = psi + 0.5 * dt * k1
        mul!(k2, H2, buf)
        k2 .*= -im
        
        @. buf = psi + 0.5 * dt * k2
        mul!(k3, H2, buf)
        k3 .*= -im
        
        @. buf = psi + dt * k3
        mul!(k4, H4, buf)
        k4 .*= -im

        # 状態ベクトルの更新を並列化
        @threads for i in 1:dim
            psi[i] += (dt / 6.0) * (k1[i] + 2k2[i] + 2k3[i] + k4[i])
        end

        if renorm
            norm_factor = sqrt(real(psi' * psi))
            @threads for i in 1:dim
                psi[i] /= norm_factor
            end
        end

        if return_traj && (s % stride == 0)
            out[:, idx] .= psi
            idx += 1
        end
    end

    return return_traj ? out : reshape(psi, :, 1)
end

function rk4_cpu_sparse_simd(
    H0::SparseMatrixCSC{ComplexF64,Int},
    mux::SparseMatrixCSC{ComplexF64,Int},
    muy::SparseMatrixCSC{ComplexF64,Int},
    Ex::Vector{Float64},
    Ey::Vector{Float64},
    psi0::Vector{ComplexF64},
    dt::Float64;
    return_traj::Bool = true,
    stride::Int = 1,
    renorm::Bool = false
)
    steps = (length(Ex) - 1) ÷ 2
    Ex3 = hcat(Ex[1:2:end-2], Ex[2:2:end-1], Ex[3:2:end])
    Ey3 = hcat(Ey[1:2:end-2], Ey[2:2:end-1], Ey[3:2:end])

    psi = copy(psi0)
    dim = length(psi)
    n_out = steps ÷ stride + 1
    out = Matrix{ComplexF64}(undef, dim, n_out)
    out[:, 1] .= psi

    # バッファの初期化（メモリアライメントを考慮）
    buf = Vector{ComplexF64}(undef, dim)
    k1 = Vector{ComplexF64}(undef, dim)
    k2 = Vector{ComplexF64}(undef, dim)
    k3 = Vector{ComplexF64}(undef, dim)
    k4 = Vector{ComplexF64}(undef, dim)
    temp = Vector{ComplexF64}(undef, dim)

    # スパース行列の準備
    rows = rowvals(H0)
    cols = Vector{Int}(undef, nnz(H0))
    for i in 1:length(rows)
        cols[i] = rows[i]
    end
    
    # 各行列から値を抽出
    H0_data = nonzeros(H0)
    mux_data = nonzeros(mux)
    muy_data = nonzeros(muy)
    
    # 3つのスパース行列を事前に確保
    H1 = copy(H0)
    H2 = copy(H0)
    H4 = copy(H0)
    
    # 値を格納するバッファ
    vals1 = Vector{ComplexF64}(undef, nnz(H0))
    vals2 = similar(vals1)
    vals4 = similar(vals1)

    idx = 2
    for s in 1:steps
        ex1, ex2, ex4 = Ex3[s, :]
        ey1, ey2, ey4 = Ey3[s, :]

        # 行列要素の更新（SIMD最適化）
        @turbo warn_check_args=false for i in eachindex(H0_data)
            vals1[i] = H0_data[i] + mux_data[i] * ex1 + muy_data[i] * ey1
            vals2[i] = H0_data[i] + mux_data[i] * ex2 + muy_data[i] * ey2
            vals4[i] = H0_data[i] + mux_data[i] * ex4 + muy_data[i] * ey4
        end
        
        # スパース行列の更新
        nonzeros(H1) .= vals1
        nonzeros(H2) .= vals2
        nonzeros(H4) .= vals4

        # 行列-ベクトル積の計算
        mul!(temp, H1, psi)
        @turbo warn_check_args=false for i in eachindex(k1)
            k1[i] = -im * temp[i]
        end
        
        @turbo warn_check_args=false for i in eachindex(buf)
            buf[i] = psi[i] + 0.5 * dt * k1[i]
        end
        
        mul!(temp, H2, buf)
        @turbo warn_check_args=false for i in eachindex(k2)
            k2[i] = -im * temp[i]
        end
        
        @turbo warn_check_args=false for i in eachindex(buf)
            buf[i] = psi[i] + 0.5 * dt * k2[i]
        end
        
        mul!(temp, H2, buf)
        @turbo warn_check_args=false for i in eachindex(k3)
            k3[i] = -im * temp[i]
        end
        
        @turbo warn_check_args=false for i in eachindex(buf)
            buf[i] = psi[i] + dt * k3[i]
        end
        
        mul!(temp, H4, buf)
        @turbo warn_check_args=false for i in eachindex(k4)
            k4[i] = -im * temp[i]
        end

        # 状態ベクトルの更新（SIMD最適化）
        @turbo warn_check_args=false for i in eachindex(psi)
            psi[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        end

        if renorm
            # ノルムの計算（SIMD最適化）
            norm_sq = zero(Float64)
            @turbo warn_check_args=false for i in eachindex(psi)
                norm_sq += abs2(psi[i])
            end
            norm_factor = sqrt(norm_sq)
            
            @turbo warn_check_args=false for i in eachindex(psi)
                psi[i] /= norm_factor
            end
        end

        if return_traj && (s % stride == 0)
            out[:, idx] .= psi
            idx += 1
        end
    end

    return return_traj ? out : reshape(psi, :, 1)
end

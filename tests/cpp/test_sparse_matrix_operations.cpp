#include <iostream>
#include <vector>
#include <complex>
#include <cassert>
#include <cmath>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;

// テスト用のスパース行列を作成
SparseMatrix<complex<double>> create_test_matrix(int dim) {
    SparseMatrix<complex<double>> matrix(dim, dim);
    vector<Triplet<complex<double>>> triplets;
    
    // 対角成分
    for (int i = 0; i < dim; ++i) {
        triplets.emplace_back(i, i, complex<double>(1.0 + 0.1 * i, 0.1 * i));
    }
    
    // 非対角成分（近接相互作用）
    for (int i = 0; i < dim - 1; ++i) {
        triplets.emplace_back(i, i + 1, complex<double>(0.1, 0.05));
        triplets.emplace_back(i + 1, i, complex<double>(0.1, -0.05)); // エルミート性
    }
    
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    return matrix;
}

// ベクトルを作成
VectorXcd create_test_vector(int dim) {
    VectorXcd vec(dim);
    for (int i = 0; i < dim; ++i) {
        vec(i) = complex<double>(1.0 / (i + 1), 0.1 / (i + 1));
    }
    return vec;
}

// 基本的なスパース行列-ベクトル積のテスト
void test_sparse_matrix_vector_product() {
    cout << "=== スパース行列-ベクトル積のテスト ===" << endl;
    
    int dim = 10;
    auto matrix = create_test_matrix(dim);
    auto vector = create_test_vector(dim);
    
    // 行列-ベクトル積の計算
    VectorXcd result = matrix * vector;
    
    // 結果の検証
    assert(result.size() == dim);
    assert(result.allFinite());
    
    cout << "✓ スパース行列-ベクトル積: 成功" << endl;
    cout << "  結果の形状: " << result.size() << endl;
    cout << "  結果の型: complex<double>" << endl;
}

// エルミート性のテスト
void test_hermitian_property() {
    cout << "=== エルミート性のテスト ===" << endl;
    
    int dim = 5;
    auto matrix = create_test_matrix(dim);
    
    // エルミート共役を計算
    SparseMatrix<complex<double>> hermitian = matrix.adjoint();
    
    // エルミート性を確認
    SparseMatrix<complex<double>> diff = matrix - hermitian;
    double max_diff = diff.coeffs().cwiseAbs().maxCoeff();
    
    cout << "最大差分: " << max_diff << endl;
    assert(max_diff < 1e-10);
    
    cout << "✓ エルミート性: 成功" << endl;
}

// 異なる次元でのテスト
void test_different_dimensions() {
    cout << "=== 異なる次元でのテスト ===" << endl;
    
    vector<int> dimensions = {5, 10, 20, 50};
    
    for (int dim : dimensions) {
        auto matrix = create_test_matrix(dim);
        auto vector = create_test_vector(dim);
        
        VectorXcd result = matrix * vector;
        
        assert(result.size() == dim);
        assert(result.allFinite());
        
        cout << "  次元 " << dim << ": 成功" << endl;
    }
    
    cout << "✓ 異なる次元でのテスト: 成功" << endl;
}

// 数値安定性のテスト
void test_numerical_stability() {
    cout << "=== 数値安定性のテスト ===" << endl;
    
    int dim = 100;
    auto matrix = create_test_matrix(dim);
    auto vector = create_test_vector(dim);
    
    // 複数回の行列-ベクトル積を計算
    for (int i = 0; i < 10; ++i) {
        VectorXcd result = matrix * vector;
        
        assert(result.allFinite());
        assert(!result.hasNaN());
        assert(!result.hasInf());
        
        // 次の反復のためにベクトルを更新
        vector = result / result.norm();
    }
    
    cout << "✓ 数値安定性: 成功" << endl;
}

// スパース性のテスト
void test_sparsity() {
    cout << "=== スパース性のテスト ===" << endl;
    
    int dim = 20;
    auto matrix = create_test_matrix(dim);
    
    // 非零要素の数を確認
    int nnz = matrix.nonZeros();
    int total_elements = dim * dim;
    double sparsity = 1.0 - static_cast<double>(nnz) / total_elements;
    
    cout << "非零要素数: " << nnz << endl;
    cout << "総要素数: " << total_elements << endl;
    cout << "スパース性: " << sparsity << endl;
    
    // スパース性が適切であることを確認
    assert(sparsity > 0.5); // 50%以上がゼロ要素
    
    cout << "✓ スパース性: 成功" << endl;
}

// 複素数演算のテスト
void test_complex_operations() {
    cout << "=== 複素数演算のテスト ===" << endl;
    
    int dim = 10;
    auto matrix = create_test_matrix(dim);
    auto vector = create_test_vector(dim);
    
    // 複素数の行列-ベクトル積
    VectorXcd result = matrix * vector;
    
    // 複素数の性質を確認
    for (int i = 0; i < result.size(); ++i) {
        complex<double> val = result(i);
        
        // 実部と虚部が有限であることを確認
        assert(isfinite(val.real()));
        assert(isfinite(val.imag()));
        
        // 複素数の絶対値が有限であることを確認
        assert(isfinite(abs(val)));
    }
    
    cout << "✓ 複素数演算: 成功" << endl;
}

// メイン関数
int main() {
    cout << "C++スパース行列操作のテスト開始" << endl;
    cout << "=================================" << endl;
    
    try {
        test_sparse_matrix_vector_product();
        test_hermitian_property();
        test_different_dimensions();
        test_numerical_stability();
        test_sparsity();
        test_complex_operations();
        
        cout << endl;
        cout << "=== すべてのテストが成功しました ===" << endl;
        return 0;
        
    } catch (const exception& e) {
        cout << "エラー: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "未知のエラーが発生しました" << endl;
        return 1;
    }
} 
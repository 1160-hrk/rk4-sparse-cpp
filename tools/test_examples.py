#!/usr/bin/env python3
"""Examples ディレクトリのテストを自動化するスクリプト"""

import sys
import os
import subprocess
import time
from pathlib import Path

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "python"

def run_example(script_name, timeout=30):
    """例のスクリプトを実行する"""
    script_path = EXAMPLES_DIR / script_name
    if not script_path.exists():
        return False, f"Script {script_name} not found"
    
    try:
        os.chdir(EXAMPLES_DIR)
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, f"Exit code: {result.returncode}\nStderr: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout} seconds"
    except Exception as e:
        return False, f"Exception: {e}"

def test_all_examples():
    """すべての例をテストする"""
    examples = [
        ("example_rk4.py", "Basic RK4 example"),
        ("two_level_excitation.py", "Two-level system excitation"),
        # ("harmonic_oscillator.py", "Harmonic oscillator dynamics"),  # 時間がかかるのでコメントアウト
    ]
    
    print("🧪 Testing Examples")
    print("=" * 50)
    
    results = []
    total_start = time.time()
    
    for script, description in examples:
        print(f"\n📝 Testing {script}: {description}")
        start_time = time.time()
        
        success, message = run_example(script)
        elapsed = time.time() - start_time
        
        if success:
            print(f"✅ PASSED ({elapsed:.2f}s)")
        else:
            print(f"❌ FAILED ({elapsed:.2f}s): {message}")
        
        results.append((script, success, elapsed, message))
    
    total_elapsed = time.time() - total_start
    
    # 結果のサマリー
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success, _, _ in results if success)
    total = len(results)
    
    for script, success, elapsed, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {script:25} ({elapsed:5.2f}s)")
        if not success and len(message) < 100:
            print(f"         └─ {message}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {total_elapsed:.2f}s")
    
    return passed == total

def test_imports():
    """インポートテストを実行"""
    print("🔍 Testing imports...")
    
    try:
        os.chdir(PROJECT_ROOT)
        sys.path.insert(0, str(PROJECT_ROOT / "python"))
        
        from excitation_rk4_sparse import rk4_cpu_sparse_py, rk4_cpu_sparse_cpp
        print("✅ Import test passed")
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """メイン関数"""
    print("🚀 Starting Examples Test Suite")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Examples dir: {EXAMPLES_DIR}")
    
    # インポートテスト
    import_success = test_imports()
    
    if not import_success:
        print("\n❌ Import tests failed. Cannot proceed with examples.")
        return False
    
    # 例のテスト
    examples_success = test_all_examples()
    
    print("\n" + "=" * 50)
    if import_success and examples_success:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
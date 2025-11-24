import time
import statistics
from pyscf import gto
from pyscf import dft as cpu_dft
from gpu4pyscf import dft as gpu_dft
from test_utils import cleanup_gpu_memory, full_cleanup

# Vitamin D3 (Cholecalciferol) coordinates extracted from PubChem (CID 5280795)
# Format: Symbol X Y Z
# Note: The original SDF had X Y Z Symbol, so we reordered them here.
vitamin_d_xyz = '''
O   -6.6365   -0.6900   -2.1167
C    1.2287    1.2867    0.2961
C    2.4804    0.4095    0.5543
C    0.2181    0.1800   -0.0951
C    1.9307   -0.8174    1.3117
C    0.4342   -0.9016    0.9680
C    1.3933    2.2852   -0.8742
C    3.6419    1.0789    1.2749
C   -1.1547    0.7401   -0.3611
C    0.8131    2.0650    1.5717
C    0.1749    2.3379   -1.8163
C   -1.1663    2.0457   -1.1304
C    4.8535    0.1405    1.4560
C    4.0843    2.3316    0.5078
C   -2.2980    0.1554    0.0256
C    5.4379   -0.4595    0.1703
C    6.5887   -1.4392    0.4214
C   -3.6009    0.7067   -0.2517
C    7.2217   -1.9863   -0.8730
C   -4.7457    0.1218    0.1352
C   -6.0970    0.7066   -0.2111
C    7.7443   -0.8761   -1.7861
C    6.1975   -2.8353   -1.6310
C   -4.7817   -1.1345    0.8771
C   -7.0565   -0.3335   -0.8012
C   -5.7286   -2.1737    0.3293
C   -7.1213   -1.5944    0.0631
C   -4.0558   -1.3497    1.9856
H    2.8229    0.0562   -0.4280
H    0.5449   -0.2659   -1.0492
H    2.4533   -1.7295    1.0031
H    2.0520   -0.7215    2.3968
H    0.1679   -1.8959    0.5934
H   -0.1650   -0.7183    1.8674
H    2.2575    2.0329   -1.4980
H    1.5625    3.3008   -0.4985
H    3.3366    1.3901    2.2816
H   -0.1478    2.5726    1.4471
H    0.7166    1.4121    2.4449
H    1.5372    2.8425    1.8325
H    0.3241    1.6047   -2.6207
H    0.1359    3.3180   -2.3066
H   -1.4081    2.8732   -0.4533
H   -1.9204    2.0291   -1.9256
H    4.5673   -0.6812    2.1247
H    5.6488    0.6833    1.9838
H    3.3861    3.1612    0.6234
H    5.0352    2.7014    0.9111
H    4.2485    2.1368   -0.5559
H   -2.2037   -0.8114    0.5004
H    5.7807    0.3615   -0.4658
H    4.6498   -0.9869   -0.3719
H    6.2326   -2.2734    1.0384
H    7.3673   -0.9309    1.0042
H   -3.6756    1.6448   -0.7913
H    8.0636   -2.6294   -0.5883
H   -5.9852    1.5337   -0.9234
H   -6.5366    1.1241    0.7040
H    6.9375   -0.2890   -2.2357
H    8.3257   -1.3074   -2.6086
H    8.4027   -0.1962   -1.2356
H    5.3857   -2.2348   -2.0531
H    6.6831   -3.3532   -2.4658
H    5.7599   -3.5994   -0.9801
H   -8.0579    0.1002   -0.8932
H   -5.3097   -2.5855   -0.5976
H   -5.8280   -3.0191    1.0214
H   -7.7294   -2.3585   -0.4352
H   -7.6060   -1.3570    1.0184
H   -4.1165   -2.2969    2.5131
H   -3.4208   -0.5914    2.4321
H   -5.7073   -0.9711   -2.0857
'''

def run_cpu_calculation(mol, verbose=0):
    """CPU並列計算を実行"""
    mf = cpu_dft.RKS(mol)
    mf.xc = 'WB97XD'
    mf.verbose = verbose
    start_time = time.time()
    energy = mf.kernel()
    elapsed_time = time.time() - start_time

    # Cleanup CPU calculation objects
    full_cleanup(mf, verbose=False)

    return energy, elapsed_time


def run_gpu_calculation(mol, verbose=0):
    """GPU計算を実行"""
    mf = gpu_dft.RKS(mol)
    mf.xc = 'WB97XD'
    mf.verbose = verbose
    start_time = time.time()
    energy = mf.kernel()
    elapsed_time = time.time() - start_time

    # Cleanup GPU calculation objects
    full_cleanup(mf, verbose=False)

    return energy, elapsed_time


def print_statistics(times, label):
    """統計情報を表示"""
    mean_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time = min(times)
    max_time = max(times)

    print(f"{label}:")
    print(f"  平均時間: {mean_time:.2f} 秒")
    print(f"  標準偏差: {stdev_time:.2f} 秒")
    print(f"  最小時間: {min_time:.2f} 秒")
    print(f"  最大時間: {max_time:.2f} 秒")
    return mean_time


def test_vitamin_d():
    print("=" * 70)
    print("ビタミンD3 (コレカルシフェロール) WB97XD/6-311G(d) ベンチマーク")
    print("GPU vs CPU 並列計算比較 (10回実行)")
    print("=" * 70)

    # Build molecule
    mol = gto.M(
        atom=vitamin_d_xyz,
        basis='6-311G(d)',
        verbose=0
    )

    print(f"原子数: {mol.natm}")
    print(f"電子数: {mol.nelectron}")
    print(f"基底関数数: {mol.nao}")
    print("-" * 70)

    # GPU warmup (初回JITコンパイル対策)
    print("GPU ウォームアップ実行中 (初回JITコンパイル)...")
    print("※ 初回は15-20分かかる場合があります")
    warmup_energy, warmup_time = run_gpu_calculation(mol, verbose=0)
    print(f"ウォームアップ完了: {warmup_time:.2f} 秒")
    print(f"エネルギー: {warmup_energy:.8f} Hartree")

    # ウォームアップ後のメモリクリーンアップ
    print("ウォームアップ後のメモリクリーンアップ中...")
    cleanup_gpu_memory(verbose=False)
    print("-" * 70)

    # GPU benchmark (10 runs)
    print("\nGPU計算ベンチマーク (10回実行)...")
    gpu_times = []
    gpu_energies = []

    for i in range(10):
        print(f"  GPU Run {i+1}/10...", end=" ", flush=True)
        energy, elapsed = run_gpu_calculation(mol, verbose=0)
        gpu_times.append(elapsed)
        gpu_energies.append(energy)
        print(f"{elapsed:.2f} 秒")

    print("-" * 70)

    # CPU benchmark (10 runs)
    print("\nCPU並列計算ベンチマーク (10回実行)...")
    cpu_times = []
    cpu_energies = []

    for i in range(10):
        print(f"  CPU Run {i+1}/10...", end=" ", flush=True)
        energy, elapsed = run_cpu_calculation(mol, verbose=0)
        cpu_times.append(elapsed)
        cpu_energies.append(energy)
        print(f"{elapsed:.2f} 秒")

    print("=" * 70)

    # 結果の統計表示
    print("\n【統計結果】")
    print("-" * 70)
    gpu_mean = print_statistics(gpu_times, "GPU計算")
    print()
    cpu_mean = print_statistics(cpu_times, "CPU並列計算")
    print("-" * 70)

    # スピードアップ率
    speedup = cpu_mean / gpu_mean
    print(f"\n【パフォーマンス比較】")
    print(f"スピードアップ率: {speedup:.2f}x (CPU/GPU)")

    if speedup > 1:
        print(f"→ GPUはCPUより {speedup:.2f}倍 高速")
    else:
        print(f"→ CPUはGPUより {1/speedup:.2f}倍 高速")

    print("-" * 70)

    # エネルギー値の比較
    print(f"\n【計算結果の妥当性チェック】")
    gpu_avg_energy = statistics.mean(gpu_energies)
    cpu_avg_energy = statistics.mean(cpu_energies)
    energy_diff = abs(gpu_avg_energy - cpu_avg_energy)

    print(f"GPU平均エネルギー: {gpu_avg_energy:.8f} Hartree")
    print(f"CPU平均エネルギー: {cpu_avg_energy:.8f} Hartree")
    print(f"差分: {energy_diff:.2e} Hartree")

    if energy_diff < 1e-6:
        print("✓ GPU/CPU計算結果は一致しています")
    else:
        print("⚠ GPU/CPU計算結果に差異があります")

    print("=" * 70)

    # 最終クリーンアップ
    print("\n【最終メモリクリーンアップ】")
    full_cleanup(mol, verbose=True)
    print("=" * 70)

if __name__ == "__main__":
    test_vitamin_d()

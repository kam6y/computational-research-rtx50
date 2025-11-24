"""
GPU Memory and Cache Management Utilities for Testing

このモジュールは、テスト実行時のGPUメモリとキャッシュの適切な管理を提供します。
"""

import gc
import sys


def cleanup_gpu_memory(verbose=True):
    """
    GPUメモリとキャッシュを適切にクリーンアップします。

    以下の処理を実行：
    1. CuPyメモリプールの解放
    2. CuPy pinnedメモリプールの解放
    3. Pythonガベージコレクションの強制実行

    Parameters
    ----------
    verbose : bool, optional
        クリーンアップの詳細を出力するかどうか（デフォルト: True）

    Returns
    -------
    dict
        解放されたメモリ情報を含む辞書（CuPyが利用可能な場合）
    """
    freed_info = {
        'cupy_available': False,
        'memory_freed': False,
        'pinned_memory_freed': False,
        'gc_collected': 0
    }

    # CuPyが利用可能か確認
    try:
        import cupy as cp
        freed_info['cupy_available'] = True

        # メモリプールの解放前の使用量を取得
        mempool = cp.get_default_memory_pool()
        used_bytes_before = mempool.used_bytes()
        total_bytes_before = mempool.total_bytes()

        # メモリプールをクリア
        mempool.free_all_blocks()
        freed_info['memory_freed'] = True

        # Pinnedメモリプールもクリア
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        freed_info['pinned_memory_freed'] = True

        # 解放後の状態
        used_bytes_after = mempool.used_bytes()
        total_bytes_after = mempool.total_bytes()

        freed_info['used_bytes_before'] = used_bytes_before
        freed_info['total_bytes_before'] = total_bytes_before
        freed_info['used_bytes_after'] = used_bytes_after
        freed_info['total_bytes_after'] = total_bytes_after
        freed_info['freed_bytes'] = total_bytes_before - total_bytes_after

        if verbose:
            print("\n[GPU Memory Cleanup]")
            print(f"  Used before: {used_bytes_before / 1024**2:.2f} MB")
            print(f"  Total before: {total_bytes_before / 1024**2:.2f} MB")
            print(f"  Freed: {freed_info['freed_bytes'] / 1024**2:.2f} MB")
            print(f"  Used after: {used_bytes_after / 1024**2:.2f} MB")
            print(f"  Total after: {total_bytes_after / 1024**2:.2f} MB")

    except ImportError:
        if verbose:
            print("\n[GPU Memory Cleanup] CuPy not available, skipping GPU memory cleanup")
    except Exception as e:
        if verbose:
            print(f"\n[GPU Memory Cleanup] Error during CuPy cleanup: {e}")

    # Pythonガベージコレクションを強制実行
    collected = gc.collect()
    freed_info['gc_collected'] = collected

    if verbose and collected > 0:
        print(f"[Garbage Collection] Collected {collected} objects")

    return freed_info


def cleanup_pyscf_objects(*objects, verbose=False):
    """
    PySCF/GPU4PySCFオブジェクトを明示的に削除します。

    Parameters
    ----------
    *objects : object
        削除するオブジェクト（mol, mf, gradなど）
    verbose : bool, optional
        削除の詳細を出力するかどうか（デフォルト: False）

    Examples
    --------
    >>> mol = gto.M(...)
    >>> mf = dft.RKS(mol)
    >>> cleanup_pyscf_objects(mol, mf)
    """
    for i, obj in enumerate(objects):
        if obj is not None:
            obj_type = type(obj).__name__
            if verbose:
                print(f"[Cleanup] Deleting {obj_type} object")
            del obj

    # オブジェクト削除後にガベージコレクション
    collected = gc.collect()
    if verbose and collected > 0:
        print(f"[Cleanup] Collected {collected} objects after deletion")


def full_cleanup(*pyscf_objects, verbose=True):
    """
    GPUメモリとPySCFオブジェクトの完全なクリーンアップを実行します。

    Parameters
    ----------
    *pyscf_objects : object
        削除するPySCFオブジェクト
    verbose : bool, optional
        クリーンアップの詳細を出力するかどうか（デフォルト: True）

    Returns
    -------
    dict
        クリーンアップ結果の情報

    Examples
    --------
    >>> mol = gto.M(...)
    >>> mf = dft.RKS(mol)
    >>> energy = mf.kernel()
    >>> full_cleanup(mol, mf)
    """
    # PySCFオブジェクトを削除
    if pyscf_objects:
        cleanup_pyscf_objects(*pyscf_objects, verbose=verbose)

    # GPUメモリをクリーンアップ
    return cleanup_gpu_memory(verbose=verbose)


def periodic_cleanup(iteration, interval=5, verbose=True):
    """
    定期的なクリーンアップを実行します（ループ内での使用を想定）。

    Parameters
    ----------
    iteration : int
        現在の反復回数
    interval : int, optional
        クリーンアップを実行する間隔（デフォルト: 5）
    verbose : bool, optional
        クリーンアップの詳細を出力するかどうか（デフォルト: True）

    Returns
    -------
    bool
        クリーンアップが実行された場合True

    Examples
    --------
    >>> for i in range(10):
    ...     energy = run_calculation()
    ...     periodic_cleanup(i, interval=5)
    """
    if iteration > 0 and iteration % interval == 0:
        if verbose:
            print(f"\n[Periodic Cleanup] Iteration {iteration}")
        cleanup_gpu_memory(verbose=verbose)
        return True
    return False


if __name__ == "__main__":
    """テスト実行"""
    print("=" * 70)
    print("GPU Memory Cleanup Utilities - Test")
    print("=" * 70)

    # 基本的なクリーンアップテスト
    print("\n1. Testing cleanup_gpu_memory()...")
    result = cleanup_gpu_memory(verbose=True)
    print(f"   Result: {result}")

    # 定期クリーンアップのシミュレーション
    print("\n2. Testing periodic_cleanup()...")
    for i in range(12):
        executed = periodic_cleanup(i, interval=5, verbose=True)
        if not executed and i % 5 != 0:
            print(f"   Iteration {i}: Skipped")

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)

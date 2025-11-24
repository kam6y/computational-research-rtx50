"""
Google Colab Auto Cleanup Extension
完全自動GPUメモリクリーンアップ for Google Colab

使い方:
    ノートブックの最初のセルで以下を実行:
    %load_ext colab_auto_cleanup

    以降、すべてのセル実行後に自動的にGPUメモリがクリーンアップされます。
"""

import sys
import gc
from IPython.core.magic import Magics, magics_class, line_magic
from IPython import get_ipython


# test_utils.pyの機能を統合（インポートできない場合に備えて）
def cleanup_gpu_memory_inline(verbose=False):
    """
    GPUメモリとキャッシュをクリーンアップ
    """
    freed_info = {
        'cupy_available': False,
        'memory_freed': False,
        'gc_collected': 0
    }

    try:
        import cupy as cp
        freed_info['cupy_available'] = True

        mempool = cp.get_default_memory_pool()
        used_before = mempool.used_bytes()
        total_before = mempool.total_bytes()

        # メモリプールをクリア
        mempool.free_all_blocks()
        freed_info['memory_freed'] = True

        # Pinnedメモリプールもクリア
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()

        total_after = mempool.total_bytes()
        freed_bytes = total_before - total_after

        if verbose:
            print(f"[Auto Cleanup] GPU Memory: {freed_bytes / 1024**2:.2f} MB freed")

    except ImportError:
        if verbose:
            print("[Auto Cleanup] CuPy not available, skipping GPU cleanup")
    except Exception as e:
        if verbose:
            print(f"[Auto Cleanup] Error: {e}")

    # ガベージコレクション
    collected = gc.collect()
    freed_info['gc_collected'] = collected

    if verbose and collected > 0:
        print(f"[Auto Cleanup] Collected {collected} objects")

    return freed_info


@magics_class
class ColabAutoCleanup(Magics):
    """
    Google Colab用自動クリーンアップマジック
    """

    def __init__(self, shell):
        super(ColabAutoCleanup, self).__init__(shell)
        self.enabled = False
        self.verbose = False
        self.cleanup_count = 0

    @line_magic
    def auto_cleanup_on(self, line):
        """自動クリーンアップを有効化"""
        self.verbose = 'verbose' in line.lower()
        self.enabled = True

        # post_run_cellイベントにフックを登録
        ip = get_ipython()
        ip.events.register('post_run_cell', self._cleanup_callback)

        mode = "verbose mode" if self.verbose else "silent mode"
        print(f"✓ Auto cleanup enabled ({mode})")
        print("  All cells will automatically clean GPU memory after execution")

    @line_magic
    def auto_cleanup_off(self, line):
        """自動クリーンアップを無効化"""
        self.enabled = False

        # フックを解除
        ip = get_ipython()
        ip.events.unregister('post_run_cell', self._cleanup_callback)

        print(f"✓ Auto cleanup disabled (cleaned {self.cleanup_count} times)")

    @line_magic
    def auto_cleanup_status(self, line):
        """現在の状態を表示"""
        status = "Enabled" if self.enabled else "Disabled"
        mode = "verbose" if self.verbose else "silent"
        print(f"Auto Cleanup Status: {status}")
        print(f"Mode: {mode}")
        print(f"Cleanup count: {self.cleanup_count}")

    def _cleanup_callback(self, result):
        """セル実行後に自動的に呼ばれるコールバック"""
        if not self.enabled:
            return

        # エラーが発生した場合はクリーンアップをスキップ
        if result.error_before_exec or result.error_in_exec:
            if self.verbose:
                print("[Auto Cleanup] Skipped due to cell execution error")
            return

        # クリーンアップ実行
        try:
            cleanup_gpu_memory_inline(verbose=self.verbose)
            self.cleanup_count += 1
        except Exception as e:
            print(f"[Auto Cleanup] Error during cleanup: {e}")


def load_ipython_extension(ipython):
    """
    IPython拡張として読み込まれたときに呼ばれる

    使い方:
        %load_ext colab_auto_cleanup
    """
    # マジックコマンドを登録
    ipython.register_magics(ColabAutoCleanup)

    # デフォルトで自動クリーンアップを有効化
    magics = ColabAutoCleanup(ipython)
    magics.enabled = True
    magics.verbose = False

    # post_run_cellイベントにフックを登録
    ipython.events.register('post_run_cell', magics._cleanup_callback)

    print("=" * 70)
    print("  GPU Auto Cleanup Extension Loaded")
    print("=" * 70)
    print("✓ Automatic GPU memory cleanup is now ENABLED")
    print("  Every cell execution will automatically free GPU memory")
    print("")
    print("Commands:")
    print("  %auto_cleanup_on [verbose]  - Enable auto cleanup")
    print("  %auto_cleanup_off            - Disable auto cleanup")
    print("  %auto_cleanup_status         - Show current status")
    print("=" * 70)


def unload_ipython_extension(ipython):
    """
    拡張がアンロードされたときに呼ばれる
    """
    # フックを解除
    try:
        magics = ColabAutoCleanup(ipython)
        ipython.events.unregister('post_run_cell', magics._cleanup_callback)
        print("✓ Auto cleanup extension unloaded")
    except:
        pass


# test_utils.pyとの互換性のための関数エイリアス
cleanup_gpu_memory = cleanup_gpu_memory_inline


if __name__ == "__main__":
    print(__doc__)

# GPU4PySCF Environment for RTX 5070 Ti on WSL2

NVIDIA RTX 5070 Ti上でGPU加速された量子化学計算（GPU4PySCF）を実行するためのDocker環境です。

## 概要

このプロジェクトは、WSL2環境でNVIDIA RTX 5070 TiのGPU性能を最大限活用し、PySCFベースの量子化学計算を高速化するための完全な開発環境を提供します。

### 主な特徴

- **NVIDIA RTX 5070 Ti最適化**: Blackwell世代GPU（compute capability 10.0）に対応
- **CUDA 12.9.1**: 最新のCUDAツールキット
- **GPU4PySCF v1.4.0**: GPU加速されたPySCF
- **完全なテストスイート**: GPU動作確認、性能比較、各種計算テスト
- **簡単セットアップ**: 自動化されたビルドと起動スクリプト

### ベースプロジェクト

[ByteDance-Seed/JoltQC](https://github.com/ByteDance-Seed/JoltQC)のDockerfileをベースに、RTX 5070 Ti向けに最適化しています。

---

## 前提条件

### ハードウェア要件
- **GPU**: NVIDIA RTX 5070 Ti (Blackwell世代)
- **RAM**: 16GB以上推奨（32GB推奨）
- **ディスク空き容量**: 20GB以上

### ソフトウェア要件
- **OS**: Windows 11 (またはWindows 10 Build 19044以降)
- **WSL2**: インストール済み
- **Docker Desktop for Windows**: インストール済み

---

## セットアップガイド

### 1. WSL2のセットアップ

#### WSL2のバージョン確認
PowerShellまたはコマンドプロンプトで以下を実行し、WSL version 2以上であることを確認してください。
```powershell
wsl --version
```

#### Ubuntuディストリビューションの確認
Ubuntuがインストールされていることを確認します。
```powershell
wsl --list --verbose
```

### 2. NVIDIA ドライバーのインストール

**重要**: WSL2でGPUを使用するには、Windows側に専用のNVIDIA GPUドライバーが必要です。

#### Windows用NVIDIA GPUドライバーのインストール
1. [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)から、GeForce RTX 50 Series > RTX 5070 Ti用のドライバーをダウンロードしてインストールします。
2. または、GeForce Experienceを使用して最新ドライバーをインストールします。

#### ドライバーの確認
PowerShellで`nvidia-smi`を実行し、GPUが認識されていることを確認してください。
また、WSL2のUbuntuターミナルでも`nvidia-smi`を実行し、同様に出力されることを確認してください。

### 3. Docker Desktopのセットアップ

1. Docker Desktopの設定（Settings）を開きます。
2. **General** > **Use the WSL 2 based engine** にチェックが入っていることを確認します。
3. **Resources** > **WSL Integration** で、使用するUbuntuディストリビューションのスイッチをONにします。
4. 設定を適用して再起動します。

#### GPUサポートの確認
WSL2ターミナルで以下を実行し、GPU情報が表示されることを確認します。
```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

---

## 環境の構築と起動

### 1. プロジェクトの準備
プロジェクトディレクトリに移動し、起動スクリプトに実行権限を付与します。

```bash
chmod +x scripts/start-environment.sh
```

### 2. 環境の起動
以下のスクリプトを実行して、環境を構築・起動します。初回はDockerイメージのビルドに時間がかかります（10-20分程度）。

```bash
./scripts/start-environment.sh
```

このスクリプトは自動的に以下を行います：
- 前提条件のチェック
- Dockerイメージのビルド
- コンテナの起動

### 3. Google Colabとの接続（オプション）
この環境をGoogle Colabのローカルランタイムとして使用する場合：

```bash
./scripts/start-colab.sh
```

---

## 使用方法

### コンテナへのアクセス

```bash
docker compose exec gpu4pyscf bash
```

### Pythonスクリプトの実行

コンテナ内で：
```bash
python3 your_script.py
```

または、コンテナ外から直接実行：
```bash
docker compose exec gpu4pyscf python3 your_script.py
```

### 環境の停止

```bash
docker compose down
```

---

## 動作確認とテスト

### テストスイートの実行
コンテナが起動したら、付属のテストスイートを実行して動作を確認します。

```bash
python3 tests/test_gpu4pyscf.py
python3 tests/test_vitamin_d.py
python3 tests/test_vitamin_d_opt.py
```

### テスト内容
1. **GPU検出**: CuPy、CUDA、GPU情報の確認
2. **GPU4PySCFインポート**: ライブラリの読み込み確認
3. **CPU/GPU DFT計算**: 水分子を用いた精度比較
4. **性能比較**: CPU vs GPUの速度比較
5. **大規模分子**: ベンゼン分子での計算テスト
6. **勾配計算**: 力の計算テスト

---

## パフォーマンスと注意点

### RTX 5070 Ti特有の挙動（初回JITコンパイル）

**重要**: RTX 5070 Ti（Blackwell世代）は非常に新しいGPUのため、**初回GPU計算時にCUDAカーネルのJITコンパイル**が発生します。

- **初回実行時**: 計算開始（`Running DFT calculation on GPU...`）で**15-20分程度停止**しているように見える場合がありますが、これは正常な動作です。
- **2回目以降**: キャッシュが効くため、数秒で完了します。
- **キャッシュの永続化**: CuPyキャッシュは永続化されているため、コンテナを再起動しても高速なままです。

### GPU加速が有効なケース
- 大規模系（100原子以上）
- 高精度基底（def2-TZVPなど）
- 多数の計算（構造最適化、MDなど）

---

## トラブルシューティング

### `nvidia-smi`がWSL2で動作しない
- Windows側でドライバーがインストールされているか確認してください。
- PowerShellで`wsl --shutdown`を実行し、WSL2を再起動してください。

### Docker GPUサポートが動作しない
- Docker DesktopのWSL Integration設定を確認してください。
- `docker run --gpus all ...` コマンドがエラーになる場合、Docker Desktopを再起動してください。

### ビルドエラー（CUDA_ARCHITECTURES）
- Dockerfileの`ENV CUDA_ARCH_LIST`が`"100"`（RTX 5070 Ti用）になっているか確認してください。

### メモリ不足（Out of Memory）
- `docker-compose.yml`の`shm_size`を増やしてください（例: `4gb`）。
- Windows側のメモリ使用状況を確認してください。

### ファイルアクセス権限エラー
- WSL2側で`chmod -R 755 .`などを実行して権限を修正してください。

---

## 技術詳細

### 構成
- **GPU**: RTX 5070 Ti (Compute Capability 10.0)
- **CUDA**: 12.9.1
- **Pythonライブラリ**: numpy, scipy, h5py, pyscf, cupy-cuda12x, cutensor-cu12, basis-set-exchange

### GPU4PySCFビルド設定
```cmake
cmake -B build -S gpu4pyscf/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCHITECTURES="100" \
    -DBUILD_LIBXC=ON
```

## ファイル構成

```
.
├── Dockerfile              # RTX 5070 Ti最適化済みDockerfile
├── docker-compose.yml      # GPU有効化設定を含むDocker Compose設定
├── scripts/
│   ├── start-environment.sh # 環境起動スクリプト
│   └── start-colab.sh       # Colab接続用スクリプト
├── tests/
│   ├── test_gpu4pyscf.py    # 包括的なテストスイート
│   └── test_cupy_cache.py   # キャッシュテスト用
├── README.md               # このファイル
└── .dockerignore           # Dockerビルド最適化
```

## 参考資料

- [PySCF Documentation](https://pyscf.org/)
- [GPU4PySCF GitHub](https://github.com/pyscf/gpu4pyscf)
- [NVIDIA CUDA on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## ライセンス

このプロジェクトは教育・研究目的で使用できます。
GPU4PySCFとPySCFのライセンスについては、それぞれのプロジェクトを参照してください。

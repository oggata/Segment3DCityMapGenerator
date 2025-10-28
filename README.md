# Segment3DCityMapGenerator-1

3D風の都市マップ画像をセグメンテーションし、結果をWebページで確認できる最小構成のリポジトリです。`base-image/` にあるベース画像からセグメンテーション結果を生成し、`output/segmentation_result.png` を `index.html` に表示します。

## 特長
- ベース画像（例: `base-image/1.png` など）からセグメンテーションを実行
- 生成結果は `output/segmentation_result.png` に保存
- `index.html` を開くだけで結果をブラウザ表示
- **壁面付き3D表示**: 建物のセグメントごとに自動的に壁面を生成し、立体的に表示
- シンプルな構成でローカル・Colab双方での実行を想定

## ディレクトリ構成
```
/Users/oggata/github-repos/Segment3DCityMapGenerator-1/
├─ base-image/                 # 入力用ベース画像
│  ├─ 1.png
│  ├─ 2.png
│  └─ 3.png
├─ output/                     # 出力（セグメンテーション結果）
│  └─ segmentation_result.png
├─ bak/                        # 旧版やバックアップ
│  └─ index.html
├─ colab_3d_city_map.py        # セグメンテーション実行用スクリプト（Colab/ローカル想定）
├─ index.html                  # 結果閲覧用シンプルなWebページ
└─ README.md                   # このファイル
```

## 必要要件
- OS: macOS (他OSでもPython環境があれば動作想定)
- Python 3.9+ 推奨
- Pythonパッケージ（想定）:
  - numpy
  - pillow
  - opencv-python
  
環境に応じて必要なパッケージを追加してください。実装内容により異なる場合があります。

### 依存関係のインストール例
```bash
python -m venv .venv
source .venv/bin/activate  # Windowsは .venv\\Scripts\\activate
pip install -U pip
pip install numpy pillow opencv-python
```

## 使い方

### 1) ローカルでセグメンテーションを実行
1. ベース画像を `base-image/` に配置（既存の `1.png` などを利用可）
2. スクリプトを実行
   ```bash
   python colab_3d_city_map.py
   ```
3. 実行後、`output/segmentation_result.png` が生成されます

> スクリプトが入力・出力パスを引数で受け取る場合は、必要に応じて下記のように指定してください（実装に合わせて適宜変更）。
> ```bash
> python colab_3d_city_map.py \
>   --input base-image/1.png \
>   --output output/segmentation_result.png
> ```

### 2) 結果をブラウザで確認
- `index.html` をダブルクリックしてブラウザで開くか、ローカルサーバを立ててアクセスします。
  ```bash
  # 任意: Pythonの簡易HTTPサーバ
  python -m http.server 8000
  # ブラウザで http://localhost:8000/index.html を開く
  ```

## Google Colabでの実行（任意）
- `colab_3d_city_map.py` の内容をColabノートブックに貼り付けるか、Colab上でリポジトリをクローンして実行します。
- 生成された画像を `output/` に保存し、必要に応じてローカルへダウンロードしてください。

## 3D壁面生成について
このプロジェクトは、セグメンテーション結果から立体的な建物を生成する際に、**自動的に壁面を追加**する機能を持っています。

### 実装方法
1. **Pythonスクリプト側（`colab_3d_city_map.py`）**:
   - 各セグメントのグリッドセルごとに、隣接セルが存在しない場合に壁面を生成
   - 上面（天井）、底面（地面）、4方向の壁面を自動的に追加
   - 壁面は元の色より少し暗い色で表現

2. **HTML側（`index.html`）**:
   - 既存のJSONデータに壁面が含まれていない場合でも、動的に壁面を生成
   - 境界エッジを検出して、上面と底面を壁面で接続
   - 後方互換性があり、古いJSONファイルでも立体表示可能

### 効果
- 建物が平面ではなく、立体的な箱型の構造として表示される
- 側面から見た際にも建物として認識しやすくなる
- 影のレンダリングがより自然になる

## パフォーマンスチューニング

### 建物の分類について
建物は5段階に分類されます：

| カテゴリ | 高さ | 説明 | 元のクラス例 |
|---------|------|------|-------------|
| 建物A（小） | 0.6 | 一戸建て、小屋 | house, shed, hut |
| 建物B（中小） | 1.0 | 店舗、小規模商業 | shop, store, market |
| 建物C（中） | 1.5 | 一般建築物、マンション | building, apartment |
| 建物D（中大） | 2.2 | オフィス、ホテル | office, hotel, stadium |
| 建物E（大） | 3.0 | 超高層ビル、タワー | skyscraper, tower, highrise |

### 高低差の調整
各カテゴリの高さは `colab_3d_city_map.py` と `index.html` の両方で定義されています：

```python
# colab_3d_city_map.py の CITY_CATEGORIES
'building_a': {'label': '建物A（小）', 'color': (255, 200, 150), 'height': 0.6, 'semantic_id': 4},
'building_b': {'label': '建物B（中小）', 'color': (255, 160, 122), 'height': 1.0, 'semantic_id': 5},
'building_c': {'label': '建物C（中）', 'color': (240, 120, 90), 'height': 1.5, 'semantic_id': 6},
'building_d': {'label': '建物D（中大）', 'color': (220, 80, 60), 'height': 2.2, 'semantic_id': 7},
'building_e': {'label': '建物E（大）', 'color': (200, 40, 40), 'height': 3.0, 'semantic_id': 8},
```

この `height` 値を変更することで高低差を調整できます。

### JSONファイルの軽量化
出力されるJSONファイルが大きすぎる場合、`colab_3d_city_map.py` の以下のパラメータを調整してください：

#### 重要なパラメータ
1. **`MIN_SEGMENT_AREA`** (デフォルト: 50)
   - 最小セグメント面積（ピクセル単位）
   - **大きくする**とセグメント数が減少し、ファイルサイズが減少
   - 推奨値: 50-200（軽量化したい場合は100-200）

2. **`MESH_RESOLUTION`** (デフォルト: 2)
   - メッシュの解像度（グリッドのステップサイズ）
   - **大きくする**と頂点数が減少し、ファイルサイズが大幅に減少
   - 推奨値: 2-4（軽量化したい場合は3-4）

3. **`MAX_IMAGE_SIZE`** (デフォルト: 1280)
   - 入力画像の最大サイズ
   - **小さくする**と全体的にセグメント数が減少
   - 推奨値: 640-1280（軽量化したい場合は640-960）

#### ファイルサイズの目安
| 設定 | ファイルサイズ | 品質 |
|------|--------------|------|
| `MESH_RESOLUTION=1`, `MIN_AREA=20` | 大（10-50MB） | 最高 |
| `MESH_RESOLUTION=2`, `MIN_AREA=50` | 中（3-15MB） | 高 |
| `MESH_RESOLUTION=3`, `MIN_AREA=100` | 小（1-5MB） | 中 |
| `MESH_RESOLUTION=4`, `MIN_AREA=200` | 最小（0.5-2MB） | 低 |

#### 調整例
```python
# 軽量化したい場合
MIN_SEGMENT_AREA = 100     # 20 → 100
MESH_RESOLUTION = 3        # 1 → 3
MAX_IMAGE_SIZE = 960       # 1280 → 960

# 品質優先の場合
MIN_SEGMENT_AREA = 20      # デフォルト
MESH_RESOLUTION = 1        # デフォルト
MAX_IMAGE_SIZE = 1280      # デフォルト
```

### セグメンテーション精度の向上
道路と建物が混ざってしまう場合、以下のパラメータを調整してください：

#### 1. タイルサイズと重複を増やす（最も効果的）
```python
TILE_SIZE = 640           # 120 → 640（大きいほど精度向上）
TILE_OVERLAP = 64         # 32 → 64（大きいほど境界が綺麗）
```

#### 2. クラス平滑化を有効にする
```python
APPLY_CLASS_SMOOTHING = True         # 道路と建物の混合を軽減
CLASS_SMOOTHING_ITERATIONS = 2       # 1-3（多いほど効果的）
```

#### 3. モルフォロジー処理を強化
```python
MORPHOLOGY_KERNEL = 7                # 5 → 7（ノイズ除去を強化）
```

#### 4. 境界の太さを調整
```python
BOUNDARY_THICKNESS = 3               # 5 → 3（細くすると混合が減る）
```

#### 5. 入力画像サイズを大きくする
```python
MAX_IMAGE_SIZE = 1280                # より大きくすると精度向上（ただし処理時間増）
```

#### 精度優先の推奨設定
```python
# 最高精度設定（処理時間は長くなる）
MAX_IMAGE_SIZE = 1600
TILE_SIZE = 640
TILE_OVERLAP = 96
MORPHOLOGY_KERNEL = 9
APPLY_CLASS_SMOOTHING = True
CLASS_SMOOTHING_ITERATIONS = 3
BOUNDARY_THICKNESS = 2
```

## トラブルシューティング
- 画像が表示されない: `output/segmentation_result.png` が存在するか、`index.html` 内の参照パスが正しいか確認してください。
- 依存パッケージのエラー: 上記のインストール例に従い、必要なパッケージを追加インストールしてください。
- 権限やパスの問題: 絶対パスでの実行や、作業ディレクトリをプロジェクト直下に合わせてから実行してください。
- JSONファイルが大きすぎる: 上記の「JSONファイルの軽量化」セクションを参照してパラメータを調整してください。
- 道路と建物が混ざる: 上記の「セグメンテーション精度の向上」セクションを参照してパラメータを調整してください。

## ライセンス
- 未設定（必要に応じて追加してください）

## 貢献
IssueやPull Requestは歓迎です。再現手順、期待結果、実際の結果を明記してください。

## 作者
- Maintainer: oggata

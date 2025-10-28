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

## トラブルシューティング
- 画像が表示されない: `output/segmentation_result.png` が存在するか、`index.html` 内の参照パスが正しいか確認してください。
- 依存パッケージのエラー: 上記のインストール例に従い、必要なパッケージを追加インストールしてください。
- 権限やパスの問題: 絶対パスでの実行や、作業ディレクトリをプロジェクト直下に合わせてから実行してください。

## ライセンス
- 未設定（必要に応じて追加してください）

## 貢献
IssueやPull Requestは歓迎です。再現手順、期待結果、実際の結果を明記してください。

## 作者
- Maintainer: oggata

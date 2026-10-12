# 東京大学「創造的ものづくりプロジェクト - AI手話翻訳プロジェクト」向け演習プログラム

本リポジトリでは，東京大学工学部「創造的ものづくりプロジェクト - AI手話翻訳プロジェクト」(大学院工業系研究科 - 創造性工学プロジェクト共通) で使用する演習プログラムを管理しています．

スクリプトの実行には，Python 3.12 と FFMPEGが必要になります．

## 演習環境の準備

### スクリプトのダウンロード

次のコマンドでリポジトリからスクリプトをダウンロードします．

``` .shell
git clone https://github.com/takayama-rado/utokyo_slt.git
```

### データセットのダウンロード

演習での説明に従って，WLASL100データセット (必要ならcustom_psjデータセットも) をダウンロードし，下記の場所に配置し，解凍してください．

``` .shell
cd <utokyo_slt_root>/src
mv <your_wlasl100.zip> ./
mv <your_custom_psj.zip> ./

unzip wlasl100.zip
unzip custom_psj.zip
```

操作後は `<utokyo_slt_root>/src/` 配下に `wlasl100` フォルダと `custom_psj` フォルダができるはずなので確認してください．

### Python 仮想環境の設定と依存モジュールのインストール

次のコマンドで Python の仮想環境を設定します．

``` .shell
# Install virtual environment.
python3 -m venv <your_venv_directory>

# Activate virtual environment.
source <your_venv_directory>/bin/activate

# Install dependencies.
cd <utokyo_slt_root>/src
pip3 install -r requirements.txt
```

なお，複数バージョンの Python がインストールされている場合は，Python 付属の `venv` モジュールではなく，`virtualenv` をモジュールを使用します．

``` .shell
# Install virtual environment.
virtualenv -p python312 <your_venv_directory>

# Other commands are the same...
```

### 環境変数の設定

次のコマンドで環境変数を設定して，スクリプト内のモジュールにアクセスできるようにします．

``` .shell
export PYTHONPATH=${PYTHONPATH}:<utokyo_slt_root>/src/modules
```

なお，`export` コマンドは一時的な設定ですので，`shell` を新たに立ち上げた場合は再設定する必要があります．永続的に設定したい場合は，システム側の環境変数設定を編集してください．

### (オプション) 孤立手話単語認識向け custom_psj の設定

`custom_psj` には，複合語などの複数単語を含むデータが含まれています．このデータセットはそのまま孤立手話単語認識の学習に使用することはできません．

下記のコマンドで，複数単語のデータを予め削除することで，孤立手話単語認識に使用することができます．

``` .shell
# 元データを削除しないようにコピー
cp -r custom_psj custom_psj_islr

# 複合語を削除
cd <utokyo_slt_root>/src/custom_psj_islr

rm ./*/*/OHAYOU_*
rm ./*/*/KONBANWA_*
rm ./*/*/KONNICHIWA_*
rm ./*/*/YOROSHIKU_*
rm ./*/*/SUMIMASEN_*
```

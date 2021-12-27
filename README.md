# ABCI_mT5

ABCIにログインします

Python3を使用できるようにモジュールのダウンロードを行います。
pip3のアップデートをします(いらないかもしれないです)

このGithubをgit cloneします

pip install -r /ABCI_mt5/requirements.txt

で必要なライブラリをダウンロードします。

mkdir data

mkdir model
でフォルダを作成します


Python3 main.py corpusファイルパスで動かします
例：Python3 main.py 

モデルのZIPファイルと結果のTSVファイルが出力されます

## オプション
- -e 数字 エポック数を指定します(earlystopは適用されます)
- --zip モデルのZipファイルのみ出力します
- --result 結果のTSVファイルのみ出力します


# ABCI_mT5

## 使用方法

1.ABCIにログイン(詳しくはSlackに載せてくださっているパワポを参照してください)
2.Python3を使用できるようにモジュールのダウンロード(こちらもパワポを参照してください)
3.(pip3のアップデート、いらないかも・・・)
4.ABCI_mT5(このGitレポジトリ)をclone
5.必要なライブラリをダウンロード(pip install -r /ABCI_mt5/requirements.txt)
6.必要なフォルダを作成(mkdir data,mkdir model)
7.main.pyを動かす



Python3 main.py corpusファイルパスで動かします
例：Python3 main.py 

モデルのZIPファイルと結果のTSVファイルが出力されます

## オプション
- コーパスファイルのパス(必須)
- -e 数字 エポック数を指定します(earlystopは適用されます)
- --zip モデルのZipファイルのみ出力します
- --result 結果のTSVファイルのみ出力します

例：  
- Python3 main.py corpus_test.txt
    - corpus_test.txtを使用した実験結果のモデルのZipファイルとTSVファイルが出力される
- Python3 main.py corpus_test.txt --zip
    - corpus_test.txtを使用した実験結果のモデルのZipファイルのみが出力される
- Python3 main.py corpus_test.txt --result
    - corpus_test.txtを使用した実験結果TSVファイルのみが出力される
- Python3 main.py corpus_test.txt -e 10
    - エポック数が10回と指定される

## うまく動かない場合
計算ノードが小さい場合があるかもしれません・・・。  
多分、パワポでは小さい計算ノードを取っていると思います。  
ポイントのこともあるので、ドカドカ使用して良いわけではないと思いますが、計算ノードの大きさを上げて
見るのも良いと思います！


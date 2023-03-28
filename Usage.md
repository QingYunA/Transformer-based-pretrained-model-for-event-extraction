# 使用文档

## 测试

```python {.line-numbers}}
 python test.py --PreTrain_Model ALBERT-base-v1  --model_path ./train_log/ALBERT-base-v1/latest_model.pt --test_path ./data/one_word.json

```
`--PreTrain_Model` 想测试的模型

`--model_path` 模型路径

`--test_path` 测试数据路径
所有的模型文件都保存在 `./train_log/` 下

## 绘图

文件路径 `./train_log/draw_fig.py`

第102行 `step = 50`

修改该值可以改变绘图的间隔

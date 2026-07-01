# Gen Runtime Assets

本目录统一保存 Gen 原始 TOF 和 Gen downstream AD 在运行时需要读取的本地资产。
这些资产属于 CausalSmartHome 主实验复现材料的一部分，不再放在代码目录里。

```text
data/gen_runtime/
  anomaly_detection_pipeline/
    attack/          下游 AD attack pkl
    check_model/     下游 AD checkpoint
    synthetic_data/  Gen 参考 synthetic pkl
    test/            下游 AD test/validation pkl
  gen_original_tof/
    check_model/     Gen 原始 TOF checkpoint
    filter_data/     Gen 原始 TOF 固定读写目录
```

对应的运行代码位于：

```text
causal_smart_home/gen_runtime/
```

该代码目录中有指向本目录的内部软链接，用来兼容 Gen 原脚本固定从相邻
`check_model`、`filter_data`、`attack`、`test` 等目录读写文件的习惯。软链接只在
本项目内部跳转，不依赖另一个 SmartGen 仓库。

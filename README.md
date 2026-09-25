# LiquidTransBLS

LiquidTransBLS 的核心训练与预测代码，用于设备退化序列的归一化剩余寿命（RUL）或退化指标预测。

## 文件结构

```text
LiquidTransBLS/
├── main.py                  # 单模型入口：加载数据、训练/增量更新、预测与 CSV 输出
├── requirements.txt         # Python 依赖
├── README.md
├── LICENSE                  # MIT 许可证
├── .gitignore               # 仅允许明确列出的源文件进入 Git
└── src/
    ├── __init__.py          # 对外导出 LiquidTransBLS
    ├── model.py             # 完整模型、特征适应、同工况更新、不确定性预测
    ├── feature_extractor.py # 完整模型使用的特征提取器
    ├── layers.py            # Liquid Cell、注意力、多尺度卷积、物理辅助头
    ├── tcbls.py             # 主算法依赖的 BLS 基础计算，并非对比实验入口
    ├── replay.py            # 分层经验回放缓存
    ├── preprocessing.py     # 归一化、滑窗、轴承特征、核心集采样、预测平滑
    └── data_loader.py       # 本地数据读取、标签构造及任务划分
```

## 安装与快速运行

建议 Python 3.11。创建环境后安装依赖：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch 自动选择可用 CUDA，否则使用 CPU；GPU 环境请安装与驱动匹配的 PyTorch 构建。
本次验证环境为 Python 3.11、NumPy 2.4.6、PyTorch 2.5.1、pandas 3.0.3、SciPy 1.17.1、scikit-learn 1.9.0；依赖范围内的所有版本组合并未逐一验证。

无需下载数据即可检查完整流程：

```bash
python main.py --dataset Synthetic --epochs 1 --d-model 8 --enhancement-nodes 16 --buffer-size 64
```

合成数据只在内存中生成，依次运行 A（初始训练）、B（同工况更新）、C（新工况适应）。此命令只检查代码能否运行，不能用于说明实际预测性能。
默认结果写入 `outputs/predictions.csv`，该目录被 Git 忽略。

真实数据示例（在仓库根目录运行）：

```bash
python main.py --dataset CMAPSS --data-root ./data --tasks A,B,C,D,E,F,G,H --epochs 10
python main.py --dataset PHM2012 --data-root ./data --tasks A,B --epochs 10
python main.py --help
```

`--data-root` 是所有数据集子目录的父目录，也可以指向仓库外的本地目录。缺少任务所需的训练或预测数据会报错，不会自动替换为合成数据；不参与当前任务的可选文件不影响运行。
`--tasks` 按加载器预定义顺序选择任务；完整增量序列应从 A 开始。单独选择后续任务会以该任务可用的训练数据初始化，不能视作完整序列的等价复现。

常用参数：`--window-size` 调整窗口长度，`--epochs` 控制特征训练轮数，`--enhancement-nodes` 控制 BLS 增强节点数，`--buffer-size` 控制回放容量，`--seed` 控制随机种子，`--output` 指定预测 CSV。
`--epochs` 在此入口始终按用户参数执行，不使用原实验脚本的任务级覆盖。默认深层维度为 64、增强节点为 3000（PHM2012 为 4000）；小规模示例参数仅用于快速验证。

## 数据集论文引用

### 1. PHM2012 Bearings（FEMTO-ST 轴承数据集）

**IEEE 格式：**

P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Chebel-Morello, N. Zerhouni, "PRONOSTIA: An experimental platform for bearings accelerated degradation tests," in Proceedings of the IEEE International Conference on Prognostics and Health Management (PHM), Denver, CO, USA, 2012.

**BibTeX：**

```bibtex
@inproceedings{nectoux2012pronostia,
  title     = {PRONOSTIA: An experimental platform for bearings accelerated degradation tests},
  author    = {Nectoux, Patrick and Gouriveau, Rafael and Medjaher, Kamal and Ramasso, Emmanuel and Chebel-Morello, Brigitte and Zerhouni, Noureddine},
  booktitle = {Proceedings of the IEEE International Conference on Prognostics and Health Management (PHM)},
  year      = {2012},
  address   = {Denver, CO, USA}
}
```

### 2. C-MAPSS Engines（NASA 涡扇发动机退化数据集）

**IEEE 格式：**

A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," in Proceedings of the 1st International Conference on Prognostics and Health Management (PHM), Denver, CO, USA, 2008. DOI: 10.1109/PHM.2008.4711414.

**BibTeX：**

```bibtex
@inproceedings{saxena2008damage,
  title     = {Damage propagation modeling for aircraft engine run-to-failure simulation},
  author    = {Saxena, Abhinav and Goebel, Kai and Simon, Don and Eklund, Neil},
  booktitle = {Proceedings of the 1st International Conference on Prognostics and Health Management (PHM)},
  year      = {2008},
  address   = {Denver, CO, USA},
  doi       = {10.1109/PHM.2008.4711414}
}
```

### 3. TJU Batteries（同济大学锂离子电池数据集）

**Nature 格式：**

Zhu, J., Wang, Y., Huang, Y. et al. Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation. Nat Commun 13, 2261 (2022). DOI: 10.1038/s41467-022-29837-w.

**BibTeX：**

```bibtex
@article{zhu2022data,
  title   = {Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation},
  author  = {Zhu, Jiangong and Wang, Yixiu and Huang, Yuan and Gopaluni, R. Bhushan and Cao, Yankai and Heere, Michael and Muhlbauer, Martin J. and Mere, Liuda and Li, Hao and Dai, Haifeng and Wei, Xuezhe},
  journal = {Nature Communications},
  volume  = {13},
  number  = {1},
  pages   = {2261},
  year    = {2022},
  doi     = {10.1038/s41467-022-29837-w}
}
```

## 数据集下载

| 数据集 | 下载链接 | 来源 |
| --- | --- | --- |
| PHM2012 Bearings（FEMTO-ST） | [下载 ZIP](https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip) | [NASA PCoE 数据仓库（FEMTO Bearing）](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| C-MAPSS Engines | [下载 ZIP](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) | [NASA PCoE 数据仓库（Turbofan Engine Degradation Simulation）](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| TJU Batteries | [Zenodo 下载页面](https://zenodo.org/records/6405084) | 论文作者发布的数据；在页面的 Files 中选择所需 ZIP 文件 |

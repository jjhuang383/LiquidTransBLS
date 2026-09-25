# LiquidTransBLS

LiquidTransBLS 的核心训练与预测代码，用于设备退化序列的归一化剩余寿命（RUL）或退化指标预测。
本仓库提取了原工程 `main.py --model Ours` 实际调用的完整模型，保留 Transformer、Liquid Cell、BLS 回归、经验回放及工况驱动的增量更新。

这里只发布源代码和使用说明。数据集、设备编号配置、预训练权重、虚拟环境、基线算法、消融实验、论文材料、运行日志和实验结果均不随仓库分发。

## 算法流程

1. 按设备分别预处理并构造滑动窗口，标签对应窗口最后一个时刻。归一化器仅在训练数据上拟合，同工况任务复用归一化器。
2. 通道与时间注意力、Transformer、多尺度卷积和 Liquid Cell 提取窗口状态及动态特征。
3. 通过增强节点和加权岭回归拟合 BLS 输出层。
4. 同工况新增数据冻结深层特征提取器，结合回放更新输出层；新工况使用训练数据和历史回放适应特征提取器，再重拟合输出层。
5. 输出窗口级预测；提供原始值以及裁剪到 `[0, 1]` 后进行 EMA 平滑的值。每个设备独立平滑。

`main.py` 采用 source-only / strict 路径，不使用测试输入或标签进行特征适应；新工况的无标签目标域统计对齐接口仍保留在模型 API 中，但主入口不调用它。
物理辅助损失不构成预测严格单调的保证。

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

## 数据集与本地目录

请自行从数据提供方获取所需数据并遵守其使用许可。这里保留的是读取接口，不附带原始文件或数据下载、格式转换脚本。

| 参数 | 数据类型 / 任务 | 加载器实际要求的本地格式 |
| --- | --- | --- |
| `CMAPSS` | C-MAPSS 航空发动机退化；A–D 为 FD001，E–H 为 FD004 | 所选任务对应的 `data/CMAPSS/train_FD001.txt` 或 `train_FD004.txt`；相应 `test_FD00x.txt`、`RUL_FD00x.txt` 为可选文件，当前轨迹预测不使用它们 |
| `PHM2012` | PHM 2012 轴承振动；A–H | `data/PHM 2012/Learning_set/Bearing*/acc_*.csv`，以及 `Full_Test_Set`（优先）或 `Test_set` |
| `XJTU` | XJTU-SY 轴承振动；三工况 A–F | `data/XJTU/35Hz12kN/Bearing1_1/Bearing1_1.npz` 等；另外两工况目录为 `37.5Hz11kN`、`40Hz10kN` |
| `IMS` | IMS 轴承振动；A–D | `data/IMS/1st_test/bearing_1.npz` 至 `bearing_4.npz` |
| `N-CMAPSS` | N-CMAPSS 发动机退化；A–F | `data/N-CMAPSS/DS01_resample_100.npz`，或本地预处理后的 `train_df.pkl` 和 `test_df.pkl` |
| `CATL` | 本地电池序列的归一化退化任务；A–F | `data/CATL/battery_data_25C.npz`、`battery_data_45C.npz`、`battery_data_60C.npz` 和私有 `tasks.json` |
| `TJU` | TJU 电池退化；A–F | `data/TJU/config_condition_based.json` 及该配置引用的 CSV 文件 |

格式要点：

- **C-MAPSS**：文本列为机组编号、循环、3 个工况变量和 21 个传感器。选取 14 个传感器，RUL 截断到 135 后除以 135。保留的任务划分使用训练文件中留出的机组 10 做整条轨迹预测，**不是官方截断测试集的基准评估**。
- **PHM2012 / XJTU / IMS**：读取振动记录后提取统计与频域特征，再添加序列特征；主入口使用线性归一化寿命标签。XJTU 的 NPZ 键为按数字排序的采样序号，每个值为一段振动矩阵；IMS 的 NPZ 包含 `data`（拼接振动矩阵）和 `timestamps`（原记录的时间戳序列）。这两种 NPZ 是预处理格式，不能直接用原始下载压缩包替代。
- **IMS**：四个任务是交叉验证式划分；本入口对每个 `initial` 任务重新创建模型和归一化器，避免不同折之间继承训练状态。
- **N-CMAPSS**：NPZ 包含 `train_data`、`test_data` 和 `var`（列名）；表的前四列按现有加载器约定为 `unit, cycle, Fc, hs`，随后 18 列为输入特征。存在 `RUL` 列时读取它，否则按每台设备的最大循环构造标签；标签再在当前所选序列内归一化。截断且没有真实 RUL 的序列不能据此获得真实剩余寿命评估。
- **CATL**：NPZ 的 `sequences` 是设备键到序列对象的字典，序列对象包含 `soh`。标签沿记录从 1 线性递减至 0，属于该加载器的归一化退化代理指标。原工程的具体设备编号已移除，需在本地 `tasks.json` 指定训练与测试设备，且两者不能重叠。
- **TJU**：配置包含 `clients` 数组；每项有 `battery_file` 和 `condition_params`，后者包含 `chemistry, temperature, charge_rate, capacity, cell_id, subgroup_id`。`battery_file` 最后两级目录用于定位本地 CSV。CSV 需要 `TJULoader.FEATURE_COLUMNS` 列出的 16 个电压/电流/充电统计特征及标签构造所需的 `capacity` 列；默认不把容量作为输入。标签依据容量阈值/退化程度构造，不等同于直接测得的剩余循环数。须提供覆盖加载器任务索引的设备清单。

CATL 本地配置示例（以下键只是占位符，应替换为自己数据中的键；不上传此文件）：

```json
{
  "A": {"temp": 25, "train": ["cell_01", "cell_02"], "test": ["cell_03"]},
  "B": {"temp": 25, "train": ["cell_04"], "test": ["cell_03"]}
}
```

完整 A–F 配置中，A/B、C/D、E/F 分别对应 25、45、60°C；A/C/E 为工况起始，B/D/F 为同工况增量任务。上述两任务配置可配合 `--tasks A,B` 使用。
含 Python 对象的 NPZ 和 pickle 数据只能从可信的本地来源加载。

## 输出与模型 API

预测 CSV 包含 `task`、`unit_index`（任务内有效设备的零起始序号）、`step`（原序列窗口末端的零起始位置）、`y_true`、`y_pred`（平滑值）和 `y_pred_raw`（原始模型输出）。所有标签/预测均采用相应加载器的归一化尺度；控制台 RMSE 也在该尺度计算。

已有预处理窗口时，可直接使用核心模型：

```python
from src import LiquidTransBLS

# X_train / X_test: (窗口数, 窗口长度, 特征数)
# y_train: (窗口数, 1)，与窗口末端对齐，采用 [0, 1] 尺度
# 多台设备应传入窗口数组列表；窗口不能跨越设备边界。
model = LiquidTransBLS(window_size=20, input_dim=14,
                      n_enhancement_nodes=3000, reg_param=0.1)
model.adapt(X_train, y_train, epochs=10, X_target=None, lambda_mmd=0.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 同工况新增的有标签窗口
model.update_data(X_new, y_new)
mean, std = model.predict_with_uncertainty(X_test, n_iter=20)
```

`predict` 不需要测试标签。MC Dropout 的 `std` 仅是采样波动量，未经置信度校准。
此精简入口每次运行从头训练，没有分发预训练模型，也没有自动保存含训练缓存的模型文件。

## 范围与限制

此仓库用于运行主算法，不包含原工程的完整实验复现流水线，也不承诺得到原论文的表格数值。主要数值计算按原 `Ours` 完整模型提取，清除了禁用分支和实验审计日志；公开的 `LiquidTransBLS` 名称指向当前完整模型。

保留的轴承预处理包含依赖整段序列长度或序列统计的操作，当前适用于离线轨迹处理，不能把本入口宣称为严格因果的实时在线预测系统。需要流式部署时，应另行设计只使用已观测前缀的预处理并重新验证。

默认 `.gitignore` 采用源文件白名单；新增模块需显式加入白名单。运行数据、私有配置和输出留在本地。

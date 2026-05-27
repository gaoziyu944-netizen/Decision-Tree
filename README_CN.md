# Robot Decision Forest

[中文](README_CN.md) | [Русский](README_RU.md)

这是一个使用 C 语言实现的机器人动作决策项目。项目通过随机森林模型，根据机器人传感器数据预测下一步动作。

模型输入包括前方距离、左侧距离、右侧距离和速度，输出机器人动作类别。

## 项目功能

- 读取 CSV 格式的训练集和测试集
- 使用 Bootstrap 采样训练多棵 CART 决策树
- 使用随机森林投票预测机器人动作
- 支持从 CSV 批量预测动作
- 支持控制台交互输入单条传感器数据并预测动作
- 支持通过命令行参数直接预测单条传感器数据
- 支持模型保存和加载
- 支持测试集评估，输出准确率、混淆矩阵、Precision、Recall 和 F1
- 支持统计特征重要性

## 项目结构

```text
Decision_Tree/
├── config.h              # 项目参数配置
├── Makefile              # 编译和运行脚本
├── README.md             # 语言切换入口
├── README_CN.md          # 中文说明文档
├── README_RU.md          # 俄语说明文档
├── data/
│   ├── train.csv         # 训练数据
│   └── test.csv          # 测试数据
├── include/
│   ├── robot_rf.h        # 随机森林接口声明
│   └── utils.h           # 工具函数接口声明
├── models/
│   └── robot_model.bin   # 训练后保存的模型文件
└── src/
    ├── main.c            # 命令行入口
    ├── robot_rf.c        # 随机森林核心实现
    └── utils.c           # 数据读取、随机数和统计工具
```

## 数据格式

CSV 文件第一行为表头，最后一列为动作标签。

```csv
dist_front,dist_left,dist_right,speed,action
63.9,2.5,27.5,22.3,2
73.6,67.7,89.2,8.7,0
```

字段说明：

- `dist_front`：机器人前方距离
- `dist_left`：机器人左侧距离
- `dist_right`：机器人右侧距离
- `speed`：机器人当前速度
- `action`：动作标签

动作标签含义：

```text
0 = FORWARD
1 = LEFT
2 = RIGHT
3 = STOP
```

## 编译项目

使用 Makefile 编译：

```bash
make
```

也可以直接使用 gcc 编译：

```bash
gcc src/main.c src/robot_rf.c src/utils.c -o robot_rf -Wall -O2 -Iinclude -I. -lm
```

在 Windows PowerShell 中，如果生成的是 `robot_rf.exe`，运行命令时可以使用：

```powershell
.\robot_rf.exe
```

## 使用方法

### 训练模型

```bash
./robot_rf train data/train.csv models/robot_model.bin
```

Windows PowerShell：

```powershell
.\robot_rf.exe train data/train.csv models/robot_model.bin
```

该命令会读取训练数据，训练随机森林模型，并把模型保存到 `models/robot_model.bin`。

### 预测动作

```bash
./robot_rf predict models/robot_model.bin data/test.csv output.csv
```

Windows PowerShell：

```powershell
.\robot_rf.exe predict models/robot_model.bin data/test.csv output.csv
```

预测结果会保存到 `output.csv`，其中包含预测动作和每个动作类别的概率。

### 交互式输入单条数据

```bash
./robot_rf input models/robot_model.bin
```

Windows PowerShell：

```powershell
.\robot_rf.exe input models/robot_model.bin
```

程序会提示你依次输入：

```text
dist_front
dist_left
dist_right
speed
```

输入完成后，程序会输出预测动作和每个动作类别的概率。

### 命令行预测单条数据

```bash
./robot_rf single models/robot_model.bin 55.9 47.9 86.9 33.3
```

Windows PowerShell：

```powershell
.\robot_rf.exe single models/robot_model.bin 55.9 47.9 86.9 33.3
```

四个数字依次表示：

```text
dist_front dist_left dist_right speed
```

该模式适合快速测试一组传感器输入，不需要准备 CSV 文件。

### 评估模型

```bash
./robot_rf evaluate models/robot_model.bin data/test.csv
```

该命令会输出：

- Overall accuracy
- Confusion Matrix
- Precision
- Recall
- F1
- Support

### 查看特征重要性

```bash
./robot_rf features models/robot_model.bin
```

该命令会统计每个特征在所有决策树中被用于分裂的次数，并计算简单的重要性比例。

## 配置参数

主要参数位于 `config.h`：

```c
#define MAX_DEPTH 10
#define N_TREES 50
#define MIN_SAMPLES_SPLIT 2
#define MIN_SAMPLES_LEAF 1
#define N_FEATURES 4
#define N_CLASSES 4
```

参数说明：

- `MAX_DEPTH`：单棵决策树最大深度
- `N_TREES`：随机森林中决策树数量
- `MIN_SAMPLES_SPLIT`：节点继续分裂所需的最小样本数
- `MIN_SAMPLES_LEAF`：叶子节点允许的最小样本数
- `N_FEATURES`：输入特征数量
- `N_CLASSES`：动作类别数量

## 实现思路

项目中的随机森林主要由以下步骤组成：

1. 从训练集中进行 Bootstrap 有放回采样
2. 每棵树训练时随机选择部分特征参与分裂
3. 使用 Gini 不纯度寻找最优分裂点
4. 多棵决策树分别预测
5. 使用多数投票得到最终动作类别

训练过程中还会使用 OOB 样本估计模型效果。

## 注意事项

- `data/train.csv` 和 `data/test.csv` 的最后一列必须是动作标签
- 如果只是预测无标签数据，当前代码仍会读取最后一列作为标签，建议保持 CSV 列格式一致
- `input` 和 `single` 模式只预测一条样本，不需要 CSV 文件
- `single` 模式必须按顺序提供 4 个传感器数值：`dist_front dist_left dist_right speed`
- 训练生成的模型文件默认保存到 `models/robot_model.bin`
- Windows 下编译产物通常为 `.exe` 文件

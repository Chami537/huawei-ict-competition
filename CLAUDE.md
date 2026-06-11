# 华为智联杯·无线程序设计大赛

2026 华为智联杯，两个子任务：
1. **AI 话务预测** — 14 天小区指标预测未来 24 小时
2. **通信资源联合分配** — 波束+资源贪心调度最大化吞吐量

## 项目结构

```
E:\华为比赛\
  main.py              ← 双模式入口（AI推理/调度）
  scheduler.py          ← Part B 调度算法
  results.csv           ← AI 预测输出（8808 行, UTF-8-SIG）
  Model/                ← AI 训练+推理代码
    preprocess.py       ← 数据加载、NIL插值、特征工程
    model.py            ← 纯 numpy 3 层 MLP
    train.py            ← 训练入口
    best_model.npz      ← 训练好的权重
    norm_params.npz     ← 标准化参数
  1780886490950118786/  ← 原始赛题数据
  example_all/          ← 四语言参考实现
```

## 运行命令

```bash
# Part A — AI推理（生成 results.csv）
python main.py

# Part A — 重新训练模型
python Model/train.py

# Part B — 调度（stdin → stdout）
python main.py < "1780886490950118786/线上阶段数据集/调度开放示例/0.in"

# 回归测试调度器
python -c "from scheduler import ...; ..."
```

## 硬约束

| 约束 | 说明 |
|------|------|
| Python 3.10.12 | 平台版本 |
| **仅 numpy 2.2.6** | 无 pandas/sklearn/pytorch |
| Part B 单用例 **<200ms** | Python 时限（C/C++ 100ms） |
| results.csv UTF-8-SIG | 含 BOM 头 |
| 标准输入输出流 | Part B 禁文件 I/O |
| 禁多线程/系统指令 | 比赛规则第7条 |
| 无第三方库 | 除 numpy 外 |

## 比赛提交

两个子任务**分别打包**：

**Part A** (`PartA_AI_Prediction.zip`): main.py + results.csv + Model/
**Part B** (`PartB_Scheduling.zip`): main.py + scheduler.py

ZIP 直接打开即为项目文件，无嵌套目录。每天最多 15 次提交。

## Git

- Remote: `origin` = https://github.com/Chami537/huawei-ict-competition (private)
- Author: Rinatsu / Rinatsuko@proton.me
- **不用 cd && git** — 用 `git -C "E:\华为比赛" <command>`
- Commit 前先问用户
- 一个功能一个 commit，不提交中间态
- 不带 Co-Authored-By: Claude

## 已知陷阱

- `preprocess.py` float 转换已加 try/except（weather 脏数据保护）
- `main.py` _next_hour 已修复为通用月份处理
- `scheduler.py` 增量 delta 计算依赖 `user_tran_cache` 一致性
- 训练随机种子固定 seed=42
- Model/ 下 .npz 文件为训练产物，推理依赖

# thinkWM — 认知负荷波动与视觉工作记忆编码

**[English](README.md) | 中文**

基于 PsychoPy 的闭环双任务实验平台，研究高认知负荷任务中投入波动对视觉工作记忆编码的影响。

## 研究背景

双任务情境中，主任务认知负荷的实时波动如何影响副任务工作记忆编码，尚缺乏在线测量手段。DeBettencourt et al. (2019) 建立了基于 RT 的闭环注意监测范式，但其主任务为低知觉负荷的简单探测。当主任务替换为高认知负荷的算术验证时，RT 波动反映的是"任务投入程度"而非"注意水平"。

根据**负荷理论**（Load Theory, Lavie 1995/2004），高认知负荷占用中央执行资源，损害对非目标刺激的主动编码。因此，高投入（快 RT / 高正确率）可能反而预测更差的 WM 表现——与 DeBettencourt 的方向相反。初步数据（5 轮重复测量 + 2 名独立被试）一致支持这一反向模式。

## 实验设计

### 双任务结构

| 任务 | 内容 | 角色 |
|------|------|------|
| 算术验证（主任务） | 判断 `a × b = ans` 是否正确，F=正确 J=错误，限时 3s | 消耗中央执行资源 |
| 颜色记忆（副任务） | 编码 6 个色块位置→颜色，3×3 九宫格鼠标回忆 | 测量 WM 编码质量 |

### 参数

- 4 blocks × 80 trials（正式模式）/ 1 block × 999 trials（debug 模式）
- 数学格式：`a × b`（a,b ∈ [2,9]，result ∈ [6,79]），真假各 50%
- 假答案 delta 限制偶数（±2, ±4, ±6），防止奇偶性快捷策略
- 算术阶段色块随机动态变化，数学阶段的色块暴露对探测编码无信息价值
- 编码时间：1.0s（注视点变黄 → 色块固定显示）
- 全程约 25-30 分钟

### 闭环触发：StateMonitor 算法

累积 z-score（基线 30 试次） + 双速 ACC EWMA，实时量化认知状态：

```
探针评分 S = |Z_RT| × (1 + min(ACC_decline / 0.12, 2.0))
```

| 状态标签 | 条件 | 认知解释 |
|----------|------|----------|
| optimal | Z_RT < -1.1, ACC=1 | 快速正确 → 深度投入 |
| cautious | Z_RT > 1.1, ACC_decline ≤ 0.03 | 慢速但正确 → 脱离/审慎 |
| lapse | Z_RT > 1.1, ACC_decline > 0.03 | 慢速且下滑 → 注意脱漏 |
| impulsive | Z_RT < -1.1, ACC=0 | 快速但错误 → 冲动 |
| acc_decline | ACC_decline > 0.12 | 正确率持续下降 |

触发条件：S ≥ 1.1 + 冷却 ≥ 5 试次 + 每 block 预算 ≤ 10 次；S ≥ 4.0 紧急超驰。

## 安装

```bash
pip install -r requirements.txt
```

核心依赖：`psychopy`, `numpy`, `scipy`。Python 3.10 推荐（详见 `.python-version`）。

分析脚本额外需 `pandas`, `matplotlib`（已包含在 requirements.txt 中）。

## 使用

```bash
# 调试模式：窗口化 1 block × 999 trials
python thinkWM.py debug

# 正式模式：全屏 4 blocks × 80 trials
python thinkWM.py
```

启动后：
1. 弹窗填写姓名和学号（仅确认身份）
2. 3 屏指导语（空格翻页）
3. 练习阶段（5 道数学 + 1 次探测后可跳过）
4. 正式实验，block 间休息 ≥ 15s
5. 自动保存并上传数据到 GitLab（内置 Deploy Token，无需配置）

## 数据上传

实验数据自动上传到 USTC GitLab Generic Packages。代码内置了 **Deploy Token**（仅 `write_package_registry` 权限——只能上传数据包，不能读写代码或修改仓库）。

**无需任何配置。** 直接运行实验，数据自动上传：

```cmd
thinkWM.exe
```

在以下位置查看数据：
```
https://git.ustc.edu.cn/YinXiran/thinkwm/-/packages
```
→ **thinkwm-data** 包 → 按日期分组（`YYYYMMDD`）

如需覆盖内置 token（如 fork 仓库后用自己的），设置环境变量：

```cmd
set GITLAB_TOKEN=你的用户名:你的token
thinkWM.exe
```

环境变量优先级高于内置 token。

## 数据输出

`data/<时间戳>/<时间戳>_explog.csv`，28 列（时间戳 = 实验启动时间）：

| 列 | 说明 |
|----|------|
| Block, Trial | 组号和试次号 |
| Trial_Type | math / probe |
| State_Label | 触发时的认知状态标签 |
| Equation | 数学等式字符串 |
| Operand_A, B | 操作数 |
| Math_Acc, Math_RT | 正确性（0/1）和反应时（s） |
| RT_Mean5, RT_SD5, RT_CV5 | 滑动 5 试次 RT 窗口统计 |
| ACC_Mean10 | 滑动 10 试次正确率 |
| RT_Micro/Meso/Macro | 三速 EWMA（α=0.30/0.10/0.02） |
| Z_RT | 累积 z-score |
| Prev_Math_Acc, Prev_RT | 前次数学试次正确性和 RT |
| Trials_Since_Probe | 距上次探测间隔试次数 |
| Is_Probe, WM_Score | 是否为探测试次，WM 得分（/6） |

## 数据分析

### 命令行（快速报告）

```bash
# 单个被试完整报告
python analysis/thinkwm_analysis.py 学号

# 多个被试对比
python analysis/thinkwm_analysis.py 学号1 学号2
```

### 交互式（Python/ipython）

```python
from analysis.thinkwm_analysis import load_subject, diff_analysis

# 加载数据
df = load_subject('学号')

# 按状态标签统计 WM
from analysis.thinkwm_analysis import wm_by_state
result = wm_by_state(df)

# 脱离态 vs 投入态差异检验
diff_analysis(df, subjects=['学号'])
```

分析功能：加载合并多 session 数据、WM 按状态分组统计、脱离/投入态差异检验、难度分层分析、post-error 分析。详细见 `analysis/thinkwm_analysis.py` 模块文档。

## 关键发现（初步数据）

1. **反向模式**：脱离态（cautious/acc_decline）WM 一致高于投入态（optimal），方向与注意范式相反
2. **难度调节**：效应由难题驱动（hard: Δ=+1.73, p=.028），easy 和 medium 不显著
3. **|Z_RT| 不独立预测 WM**：r = -0.40 ~ 0.07
4. **天花板效应**：高 WM 被试（009: 5.27/6）无状态区分度
5. **Post-error 无 WM 影响**：Δ=+0.006, p=.73，但状态标签偏移（χ²=36.8, p<.0001）

见 `docs/latex/paper_outline.tex` 和 `analysis/` 获取完整分析。

## 快捷键

| 键 | 功能 |
|----|------|
| F | 等式正确 |
| J | 等式错误 |
| ESC | 退出实验（需确认） |
| 空格 | 翻页 / 跳过练习 |

## 测试

```bash
python -m pytest tests/ -v    # 45 项，~1 秒
```

- **StateMonitor 测试**（32 项）：初始化、z-score 计算、ACC EWMA、状态标签分类、触发逻辑、边界情况
- **数学题生成测试**（13 项）：操作数范围、真假答案、假答案偶性校验、格式验证

## 技术栈

- Python 3.10 + PsychoPy + NumPy + SciPy
- 自动化测试：pytest 45 项（StateMonitor 32 项 + 数学题生成 13 项）
- MockPsychopy 测试隔离

## 项目结构

```
thinkWM.py          — 主实验程序
build_exe.py        — PyInstaller 构建脚本
README.md           — 本文件
data/               — 实验数据
analysis/           — 分析脚本
```

## 引用

相关论文：DeBettencourt, M. T., Keene, P. A., Awh, E., & Vogel, E. K. (2019). Real-time monitoring of attention fluctuations in the visual working memory system. *Nature Human Behaviour*, 3(8), 792–800.

## 免责声明

本项目为**自主研究**，独立完成。未经任何导师指导、评审或机构认可。实验设计、分析与结论仅代表作者个人工作，可能存在错误或疏漏。**未经同行评审。** 请审慎看待研究结果。

## 许可

MIT License，详见 [LICENSE](LICENSE)。

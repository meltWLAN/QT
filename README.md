# 超神系统 - 中国股市量子预测引擎

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**超神系统**是一个基于量子纠缠理论的中国股市预测系统，集成了政策量子场、板块轮动共振器、北向资金探测器等多个先进算法模块，旨在为投资者提供科学、全面的市场分析和投资建议。

![超神系统](docs/images/supergod_banner.png)

## 核心功能

* **市场整体分析**：基于量子算法分析大盘走势、风险评估和热点板块轮动
* **政策影响评估**：政策量子场模型实时捕捉政策变化对市场的影响
* **北向资金分析**：量化分析北向资金流向及其对个股的偏好
* **板块轮动预测**：预测下一轮可能出现的热点板块，把握轮动机会
* **个股精准推荐**：综合多维度因素，生成个股买卖建议
* **风险预警系统**：实时监控市场风险，提供预警信息
* **自适应仓位管理**：根据市场风险水平自动调整建议仓位

## 安装指南

### 系统要求

* Python 3.8+
* 50MB以上磁盘空间
* 网络连接（用于获取市场数据）

### 安装步骤

1. 克隆代码库到本地

```bash
git clone https://github.com/yourusername/supergod-system.git
cd supergod-system
```

2. 创建并激活虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖包

```bash
pip install -r requirements.txt
```

4. 初始化系统（创建缓存目录等）

```bash
python setup.py
```

## 快速开始

运行超神系统主程序：

```bash
python run_supergod_system.py
```

查看股票推荐（指定关注的股票列表）：

```bash
python run_supergod_system.py --stocks 600519,000858,601318,600036,000333
```

添加政策事件（用于评估政策影响）：

```bash
python run_supergod_system.py --policy --policy-type 降息 --policy-strength 0.8 --policy-sectors 银行,房地产 --policy-desc "央行降息25个基点"
```

## 配置说明

超神系统支持通过配置文件进行高级配置，创建`config.json`文件：

```json
{
  "cache_dir": "./cache/china_market",
  "quantum_dimensions": 8,
  "learning_rate": 0.01,
  "entanglement_factor": 0.3,
  "policy_influence_weight": 0.4,
  "north_fund_weight": 0.3,
  "sector_rotation_weight": 0.3,
  "default_stocks": ["600519", "000858", "601318", "600036", "000333"],
  "watched_sectors": ["银行", "医药", "食品饮料", "新能源", "半导体", "军工", "房地产"],
  "data_update_interval": 60,
  "risk_threshold": 0.7,
  "max_position_high_risk": 0.2,
  "max_position_medium_risk": 0.5,
  "max_position_low_risk": 0.8
}
```

然后使用配置文件运行系统：

```bash
python run_supergod_system.py --config config.json
```

## 系统架构

超神系统由以下核心模块组成：

1. **量子纠缠引擎** - 系统的核心，负责量子态预测
2. **中国股市数据源** - 负责获取和处理A股市场数据
3. **政策量子场** - 捕捉政策变化对市场的量子影响
4. **板块轮动共振器** - 检测和预测行业板块轮动
5. **北向资金探测器** - 分析北向资金流向及个股偏好
6. **风险分析器** - 多维度评估市场和个股风险
7. **策略生成器** - 综合各模块输出，生成投资建议

## 输出解读

超神系统会在终端输出分析结果摘要，同时将详细结果保存到JSON文件中。

### 终端输出示例

```
==================================================
超神系统 - 市场预测与投资建议
==================================================

市场状况: 中等风险
整体风险: 0.45
风险趋势: 震荡

主要指数:
  上证指数: 3452.28 (+0.86%)
  深证成指: 11502.36 (+1.25%)
  创业板指: 2450.21 (+1.85%)

北向资金: 15.26亿 (5日)
资金趋势: 加速流入

热点板块:
  新能源
  半导体
  医药

下一轮潜在热点:
  军工
  银行
  食品饮料

投资组合建议:
建议仓位: 50%

板块配置:
  新能源: 12.5%
  半导体: 10.0%
  医药: 7.5%
  军工: 7.5%
  银行: 5.0%

个股推荐:
  贵州茅台(600519): 强烈买入, 配置5.0%
  格力电器(000651): 买入, 配置3.0%
  招商银行(600036): 小仓位买入, 配置1.0%
  比亚迪(002594): 北向资金流入, 配置2.0%
  隆基股份(601012): 买入, 配置2.5%

风险提示:
  行业新能源风险较高(0.75)，建议减少该行业持仓
  政策不确定性风险较高(0.72)，建议关注政策动向
  个股002594风险较高(0.78)，建议减持或观望

==================================================
分析时间: 2023-09-25 15:30:45
==================================================
```

### JSON输出

系统会在`cache/china_market/results`目录下生成多个JSON文件：

- `prediction_YYYYMMDD_HHMMSS.json` - 详细预测结果
- `portfolio_YYYYMMDD_HHMMSS.json` - 投资组合建议
- `recommendation_YYYYMMDD_HHMMSS.json` - 个股推荐

## 高级用法

### 1. 定制量子纠缠维度

增加量子维度可提高预测精度，但会增加计算资源消耗：

```json
{
  "quantum_dimensions": 12,
  "entanglement_factor": 0.4
}
```

### 2. 风险偏好调整

调整风险阈值和仓位配置，适应不同风险偏好：

```json
{
  "risk_threshold": 0.6,
  "max_position_high_risk": 0.3,
  "max_position_medium_risk": 0.6,
  "max_position_low_risk": 0.9
}
```

### 3. 行业权重定制

重点关注特定行业：

```json
{
  "watched_sectors": ["新能源", "半导体", "医药", "消费"],
  "sector_rotation_weight": 0.4
}
```

## 常见问题

**Q: 如何获取最新数据？**  
A: 系统会自动获取最新市场数据。如需强制更新，可删除cache目录后重新运行系统。

**Q: 北向资金数据更新频率？**  
A: 北向资金数据每日收盘后更新一次，盘中数据为实时估算值。

**Q: 如何评估政策影响？**  
A: 使用`--policy`参数添加新的政策事件，系统会自动评估其对市场和板块的影响。政策影响会随时间逐渐衰减。

**Q: 系统预测周期？**  
A: 系统默认提供短期（1-3日）和中期（5-10日）预测，暂不支持长期预测。

## 免责声明

超神系统仅供学习研究使用，投资有风险，入市需谨慎。系统提供的分析和建议不构成投资建议，用户需自行承担投资风险。系统开发者不对因使用本系统而导致的任何损失负责。

## 联系方式

有任何问题或建议，请通过以下方式联系：

- 邮箱：support@supergod-system.com
- 网站：https://www.supergod-system.com
- GitHub Issues: https://github.com/yourusername/supergod-system/issues

## 更新日志

**v1.0.0 (2023-09-25)**
- 首次发布
- 实现核心功能模块
- 支持A股市场分析

**v1.1.0 (计划中)**
- 增加港股分析模块
- 优化量子引擎性能
- 新增板块资金流向分析

## 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

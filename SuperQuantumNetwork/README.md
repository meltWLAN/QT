# 超神系统 - 中国市场分析模块

超神系统中国市场分析模块是专为桌面版设计的高级市场分析组件，提供实时市场数据展示、板块轮动分析、个股推荐和量子AI预测功能。

## 功能特点

- **市场实时数据显示**：展示上证指数、深证成指、创业板指等核心指数的实时数据和涨跌情况
- **热点板块分析**：展示当前市场热点板块和预测下一轮可能的热点板块
- **北向资金跟踪**：实时监控北向资金流向，掌握外资动向
- **量子AI预测**：采用前沿量子算法分析市场趋势，科学评估市场风险
- **智能个股推荐**：基于行业热点和风险评估推荐潜力个股
- **投资策略建议**：根据市场情况自动调整仓位建议，辅助投资决策

## 系统架构

该模块采用MVC架构设计，主要包含以下组件：

1. **数据源层 (data_sources.py)**：负责从各种渠道获取市场数据，支持实时数据和缓存机制
2. **控制器层 (market_controllers.py)**：处理市场数据的分析和预测逻辑，连接数据源和视图
3. **AI引擎 (quantum_ai.py)**：基于量子计算概念的高级分析引擎，提供市场趋势和板块轮动预测
4. **视图层 (china_market_view.py)**：市场数据可视化界面，基于PyQt5实现，提供直观的用户交互

## 安装要求

- Python 3.7+
- PyQt5
- NumPy
- pandas

## 使用方法

### 基本导入

```python
# 导入模块
from SuperQuantumNetwork import initialize_market_module

# 初始化（在应用主窗口中）
initialize_market_module(main_window, tab_widget)
```

### 手动创建市场视图

```python
from SuperQuantumNetwork import create_market_view, MarketDataController

# 创建控制器
controller = MarketDataController()

# 创建视图
create_market_view(main_window, tab_widget, controller)
```

### 使用AI预测引擎

```python
from SuperQuantumNetwork import get_market_prediction, get_index_data

# 获取市场数据
sh_index = get_index_data('000001.SH')
sz_index = get_index_data('399001.SZ')
cyb_index = get_index_data('399006.SZ')

# 构建市场数据字典
market_data = {
    'sh_index': sh_index,
    'sz_index': sz_index,
    'cyb_index': cyb_index
}

# 获取预测
prediction = get_market_prediction(market_data)
print(prediction)
```

## 配置选项

系统提供多种配置选项，可通过传递配置字典来自定义行为：

```python
# AI引擎配置
ai_config = {
    'risk_sensitivity': 0.7,  # 风险敏感度 (0-1)
    'trend_factor': 1.0,      # 趋势因子
    'noise_reduction': 0.6,   # 噪声减少因子
    'quantum_factor': 0.8     # 量子因子
}

# 数据源配置
data_config = {
    'use_cache': True,        # 是否使用缓存
    'cache_expiry': 3600,     # 缓存过期时间(秒)
    'cache_dir': '~/超神系统/cache'  # 缓存目录
}

# 创建控制器时传入配置
controller = MarketDataController(config={
    'ai_config': ai_config,
    'data_dir': '~/超神系统/market_data'
})
```

## 开发扩展

系统设计支持灵活扩展，您可以通过以下方式进行定制：

1. **添加新数据源**：扩展 `data_sources.py` 以支持更多数据获取渠道
2. **增强AI模型**：在 `quantum_ai.py` 中添加更复杂的分析模型
3. **自定义界面**：扩展 `china_market_view.py` 以添加更多可视化组件

## 注意事项

- 本模块仅用于市场分析和参考，不构成投资建议
- 在首次使用时，系统会自动创建必要的目录和配置文件
- 当实际数据源不可用时，系统会自动切换到模拟数据模式
- 日志文件保存在 `~/超神系统/logs/` 目录下

## 许可证

版权所有 © 2023 超神系统开发团队 
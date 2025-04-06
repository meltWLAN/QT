"""
高级可视化模块 - 实现复杂的数据可视化和交互式图表
提供交易决策点标注、市场相关性热力图和实时决策过程可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import logging
import os
import io
import base64
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# 设置日志
logger = logging.getLogger(__name__)

# 设置可视化风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_style("darkgrid")

# 定义自定义颜色映射
quantum_cmap = LinearSegmentedColormap.from_list(
    "quantum", 
    [(0.0, "#3498db"),   # 蓝色 - 低值
     (0.5, "#ecf0f1"),   # 白色 - 中值
     (1.0, "#e74c3c")]   # 红色 - 高值
)

# 量子化风格的蓝色和红色配色
BUY_COLOR = "#3498db"  # 蓝色
SELL_COLOR = "#e74c3c"  # 红色
HOLD_COLOR = "#7f8c8d"  # 灰色

class AdvancedChartingEngine:
    """高级图表引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化高级图表引擎
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.figure_size = self.config.get("figure_size", (14, 8))
        self.dpi = self.config.get("dpi", 100)
        self.theme = self.config.get("theme", "dark")
        self.max_points = self.config.get("max_points", 500)
        self.show_volume = self.config.get("show_volume", True)
        
        # 设置主题
        self._set_theme()
        
    def _set_theme(self):
        """设置图表主题"""
        if self.theme == "dark":
            plt.style.use("dark_background")
            self.bg_color = "#121212"
            self.text_color = "#f0f0f0"
            self.grid_color = "#333333"
        else:
            plt.style.use("ggplot")
            self.bg_color = "#f5f5f5"
            self.text_color = "#333333"
            self.grid_color = "#dddddd"
            
    def plot_performance_with_decisions(self, 
                                       performance: Dict[str, Any],
                                       trade_decisions: List[Dict[str, Any]],
                                       save_path: str = None) -> str:
        """绘制包含交易决策点的性能图表
        
        Args:
            performance: 性能数据
            trade_decisions: 交易决策列表
            save_path: 保存路径
            
        Returns:
            str: 图表路径或base64编码
        """
        # 提取性能数据
        dates = performance.get("dates", [])
        values = performance.get("value_history", [])
        
        if not dates or not values or len(dates) != len(values):
            logger.error("性能数据无效或不完整")
            return None
            
        # 转换日期
        try:
            if isinstance(dates[0], str):
                dates = [datetime.strptime(d, "%Y-%m-%d") if "-" in d else
                         datetime.strptime(d, "%Y%m%d") for d in dates]
        except Exception as e:
            logger.error(f"日期转换失败: {e}")
            
        # 创建DataFrame
        df = pd.DataFrame({
            "date": dates,
            "value": values
        })
        
        # 计算收益率
        if "return_history" not in performance and len(values) > 1:
            returns = [values[i] / values[i-1] - 1 for i in range(1, len(values))]
            returns.insert(0, 0)  # 第一天收益率为0
            df["return"] = returns
            df["cum_return"] = np.cumprod(1 + np.array(returns)) - 1
        else:
            df["return"] = performance.get("return_history", [0] * len(dates))
            df["cum_return"] = performance.get("cumulative_return_history", 
                                             [v/values[0] - 1 for v in values])
            
        # 提取决策信息，并转换为DataFrame
        decision_dates = []
        decision_values = []
        decision_types = []
        decision_conf = []
        
        for decision in trade_decisions:
            if "timestamp" in decision and "action" in decision:
                try:
                    # 转换决策时间戳
                    if isinstance(decision["timestamp"], str):
                        d_time = datetime.fromisoformat(decision["timestamp"].replace("Z", "+00:00"))
                    else:
                        d_time = datetime.fromtimestamp(decision["timestamp"])
                        
                    # 找到最近的价格点
                    idx = min(range(len(dates)), key=lambda i: abs((dates[i] - d_time).total_seconds()))
                    
                    decision_dates.append(dates[idx])
                    decision_values.append(values[idx])
                    decision_types.append(decision["action"])
                    decision_conf.append(decision.get("confidence", 0.5))
                except Exception as e:
                    logger.warning(f"决策转换错误: {e}")
                    
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=self.figure_size, dpi=self.dpi, 
                               gridspec_kw={"height_ratios": [3, 1, 1]})
        
        # 1. 资产价值图
        ax1 = axes[0]
        ax1.plot(df["date"], df["value"], linewidth=2, color="#2ecc71")
        ax1.set_title("Portfolio Value Evolution", fontsize=16)
        ax1.set_ylabel("Portfolio Value", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加决策点
        for date, value, d_type, conf in zip(decision_dates, decision_values, decision_types, decision_conf):
            if d_type == "buy":
                color = BUY_COLOR
                marker = "^"
            elif d_type == "sell":
                color = SELL_COLOR
                marker = "v"
            else:
                color = HOLD_COLOR
                marker = "o"
                
            # 根据置信度调整标记大小
            size = 50 + conf * 100
            ax1.scatter(date, value, color=color, s=size, zorder=10, marker=marker,
                       alpha=0.8, edgecolors="white", linewidth=1)
            
        # 2. 累积收益率
        ax2 = axes[1]
        ax2.plot(df["date"], df["cum_return"] * 100, linewidth=2, color="#3498db")
        ax2.set_ylabel("Cumulative Return (%)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 填充正收益区域
        ax2.fill_between(df["date"], df["cum_return"] * 100, 0, 
                        where=(df["cum_return"] >= 0), color="#2ecc71", alpha=0.3)
        ax2.fill_between(df["date"], df["cum_return"] * 100, 0, 
                        where=(df["cum_return"] < 0), color="#e74c3c", alpha=0.3)
        
        # 3. 日收益率
        ax3 = axes[2]
        ax3.bar(df["date"], df["return"] * 100, color=np.where(df["return"] >= 0, "#2ecc71", "#e74c3c"),
               alpha=0.7)
        ax3.set_ylabel("Daily Return (%)", fontsize=12)
        ax3.set_xlabel("Date", fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 日期格式化
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
        # 调整布局
        plt.tight_layout()
        fig.autofmt_xdate()
        
        # 添加性能指标
        metrics_text = (
            f"Total Return: {performance.get('total_return', 0)*100:.2f}%  |  "
            f"Sharpe: {performance.get('sharpe', 0):.2f}  |  "
            f"Max Drawdown: {performance.get('max_drawdown', 0)*100:.2f}%  |  "
            f"Win Rate: {performance.get('win_rate', 0)*100:.1f}%  |  "
            f"Trades: {performance.get('trade_count', 0)}"
        )
        fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=12, 
                transform=fig.transFigure)
        
        # 保存或返回图表
        if save_path:
            try:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"性能图表已保存至: {save_path}")
                return save_path
            except Exception as e:
                logger.error(f"保存图表失败: {e}")
                return None
        else:
            # 返回base64编码的图片
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_str
        
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                save_path: str = None) -> str:
        """绘制相关性热力图
        
        Args:
            correlation_matrix: 相关性矩阵
            save_path: 保存路径
            
        Returns:
            str: 图表路径或base64编码
        """
        if correlation_matrix is None or correlation_matrix.empty:
            logger.error("相关性矩阵为空")
            return None
            
        # 创建图表
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # 绘制热力图
        mask = np.zeros_like(correlation_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True  # 只显示下三角
        
        sns.heatmap(correlation_matrix, mask=mask, cmap=quantum_cmap, vmax=1.0, vmin=-1.0, 
                  center=0, annot=True, fmt=".2f", linewidths=0.5, ax=ax, 
                  cbar_kws={"shrink": .8})
        
        ax.set_title("Asset Correlation Heatmap", fontsize=16)
        
        # 保存或返回图表
        if save_path:
            try:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"相关性热力图已保存至: {save_path}")
                return save_path
            except Exception as e:
                logger.error(f"保存图表失败: {e}")
                return None
        else:
            # 返回base64编码的图片
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_str
            
    def plot_risk_contribution(self, risk_data: Dict[str, Any],
                              save_path: str = None) -> str:
        """绘制风险贡献图
        
        Args:
            risk_data: 风险数据
            save_path: 保存路径
            
        Returns:
            str: 图表路径或base64编码
        """
        if not risk_data or "risk_contribution" not in risk_data:
            logger.error("风险数据无效或不完整")
            return None
            
        risk_contrib = risk_data["risk_contribution"]
        if not risk_contrib:
            logger.error("风险贡献数据为空")
            return None
            
        # 排序风险贡献
        symbols = []
        contributions = []
        
        for symbol, contrib in sorted(risk_contrib.items(), key=lambda x: x[1], reverse=True):
            symbols.append(symbol)
            contributions.append(contrib)
            
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        # 绘制条形图
        bars = ax.barh(symbols, contributions, color=plt.cm.viridis(np.linspace(0, 1, len(symbols))))
        
        # 添加数据标签
        for i, (symbol, contrib) in enumerate(zip(symbols, contributions)):
            ax.text(contrib + 0.01, i, f"{contrib:.2%}", va='center')
            
        ax.set_title("Portfolio Risk Contribution", fontsize=16)
        ax.set_xlabel("Risk Contribution (%)", fontsize=12)
        ax.set_ylabel("Asset", fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加总体风险指标
        risk_level = risk_data.get("portfolio_risk", 0)
        div_score = risk_data.get("diversification_score", 0)
        
        metrics_text = (
            f"Total Risk: {risk_level:.2%}  |  "
            f"Diversification Score: {div_score:.2f}  |  "
            f"Status: {risk_data.get('risk_status', 'unknown')}"
        )
        
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=12)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 保存或返回图表
        if save_path:
            try:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"风险贡献图已保存至: {save_path}")
                return save_path
            except Exception as e:
                logger.error(f"保存图表失败: {e}")
                return None
        else:
            # 返回base64编码的图片
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_str

    def plot_quantum_decision_process(self, quantum_states: Dict[str, Dict[str, Any]],
                                    decision_history: List[Dict[str, Any]],
                                    save_path: str = None) -> str:
        """绘制量子决策过程可视化
        
        Args:
            quantum_states: 量子状态
            decision_history: 决策历史
            save_path: 保存路径
            
        Returns:
            str: 图表路径或base64编码
        """
        if not quantum_states or not decision_history:
            logger.error("量子状态或决策历史数据无效")
            return None
            
        # 创建图表
        fig = plt.figure(figsize=(15, 10), dpi=self.dpi)
        
        # 1. 主量子态概率图 (上方)
        grid = plt.GridSpec(3, 4, figure=fig)
        ax_main = fig.add_subplot(grid[0:2, :])
        
        # 提取主要市场分段的量子态概率历史
        main_segment = list(quantum_states.keys())[0]  # 使用第一个市场分段
        
        # 提取决策历史中的概率数据
        steps = list(range(len(decision_history)))
        buy_probs = []
        sell_probs = []
        hold_probs = []
        
        for decision in decision_history:
            if "probabilities" in decision:
                probs = decision["probabilities"]
                buy_probs.append(probs.get("buy", 0))
                sell_probs.append(probs.get("sell", 0))
                hold_probs.append(probs.get("hold", 0))
            else:
                # 没有概率数据，使用默认值
                buy_probs.append(0)
                sell_probs.append(0)
                hold_probs.append(1)  # 默认hold
                
        # 绘制概率演化
        ax_main.plot(steps, buy_probs, 'b-', label='Buy', linewidth=2)
        ax_main.plot(steps, sell_probs, 'r-', label='Sell', linewidth=2)
        ax_main.plot(steps, hold_probs, 'g-', label='Hold', linewidth=2)
        ax_main.set_title(f"Quantum State Probability Evolution - {main_segment}", fontsize=16)
        ax_main.set_ylabel("Probability", fontsize=12)
        ax_main.set_ylim(0, 1)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='upper right')
        
        # 标记决策点
        for i, decision in enumerate(decision_history):
            action = decision.get("action", "hold")
            if action != "hold":
                marker = "^" if action == "buy" else "v"
                color = BUY_COLOR if action == "buy" else SELL_COLOR
                ax_main.scatter(i, decision["probabilities"].get(action, 0.5), 
                            color=color, s=100, zorder=10, marker=marker)
                
        # 2. 量子相位图 (左下)
        ax_phase = fig.add_subplot(grid[2, 0:2], polar=True)
        
        # 提取最近的量子路径积分信息
        recent_path = None
        for decision in reversed(decision_history):
            if "quantum_path_integral" in decision:
                recent_path = decision["quantum_path_integral"]
                break
                
        if recent_path:
            # 绘制量子相位
            theta = np.linspace(0, 2*np.pi, 100)
            radius = np.ones_like(theta) * recent_path.get("magnitude", 0.5)
            
            ax_phase.plot(theta, radius, color="#9b59b6", alpha=0.7)
            # 标记当前相位
            phase_angle = recent_path.get("phase", 0)
            ax_phase.scatter(phase_angle, recent_path.get("magnitude", 0.5), 
                           color="#e74c3c", s=100, zorder=10)
                           
            ax_phase.set_title("Quantum Phase Representation", fontsize=12)
        else:
            ax_phase.set_title("Quantum Phase (No Data)", fontsize=12)
            
        # 3. 决策类型分布 (右下)
        ax_types = fig.add_subplot(grid[2, 2:])
        
        # 计算决策类型分布
        decision_types = [d.get("decision_type", "unknown") for d in decision_history]
        unique_types = set(decision_types)
        type_counts = {t: decision_types.count(t) for t in unique_types}
        
        # 绘制决策类型饼图
        if type_counts:
            labels = list(type_counts.keys())
            sizes = list(type_counts.values())
            explode = [0.1 if t == "collapsed" else 0 for t in labels]  # 突出量子坍缩
            
            # 使用量子风格的颜色
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(len(labels))]
            
            wedges, texts, autotexts = ax_types.pie(
                sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90
            )
            
            # 设置字体大小
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                
            ax_types.set_title("Decision Type Distribution", fontsize=12)
        else:
            ax_types.text(0.5, 0.5, "No Decision Type Data", 
                       ha='center', va='center', fontsize=12)
            ax_types.set_title("Decision Type Distribution", fontsize=12)
            
        # 调整布局
        plt.tight_layout()
        
        # 保存或返回图表
        if save_path:
            try:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"量子决策过程图已保存至: {save_path}")
                return save_path
            except Exception as e:
                logger.error(f"保存图表失败: {e}")
                return None
        else:
            # 返回base64编码的图片
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_str
            
    def create_dashboard(self, performance_data: Dict[str, Any],
                       risk_data: Dict[str, Any],
                       decision_history: List[Dict[str, Any]],
                       correlation_matrix: pd.DataFrame = None,
                       quantum_states: Dict[str, Dict[str, Any]] = None,
                       save_dir: str = None) -> Dict[str, str]:
        """创建完整的交易仪表盘
        
        Args:
            performance_data: 性能数据
            risk_data: 风险数据
            decision_history: 决策历史
            correlation_matrix: 相关性矩阵
            quantum_states: 量子状态
            save_dir: 保存目录
            
        Returns:
            dict: 图表路径字典
        """
        charts = {}
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # 生成性能图表
        if performance_data:
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, "performance_chart.png")
            charts["performance"] = self.plot_performance_with_decisions(
                performance_data, decision_history, save_path
            )
            
        # 生成风险贡献图
        if risk_data:
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, "risk_chart.png")
            charts["risk"] = self.plot_risk_contribution(risk_data, save_path)
            
        # 生成相关性热力图
        if correlation_matrix is not None:
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, "correlation_heatmap.png")
            charts["correlation"] = self.plot_correlation_heatmap(correlation_matrix, save_path)
            
        # 生成量子决策过程图
        if quantum_states and decision_history:
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, "quantum_decision_process.png")
            charts["quantum"] = self.plot_quantum_decision_process(quantum_states, decision_history, save_path)
            
        return charts 
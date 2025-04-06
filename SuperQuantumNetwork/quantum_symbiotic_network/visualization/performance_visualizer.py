#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能可视化模块
提供交易系统性能的可视化功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PerformanceVisualizer")

class PerformanceVisualizer:
    """性能可视化工具，提供多种图表展示交易系统性能"""
    
    def __init__(self, config=None, output_dir="charts"):
        """
        初始化可视化工具
        
        参数:
            config (dict, 可选): 可视化配置项
            output_dir (str): 图表输出目录
        """
        self.config = config or {}
        self.output_dir = output_dir
        
        # 设置Seaborn样式
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"性能可视化工具已初始化，图表将保存至: {self.output_dir}")
    
    def plot_portfolio_performance(self, portfolio_df, title="投资组合表现", filename="portfolio_performance.png"):
        """
        绘制投资组合表现图
        
        参数:
            portfolio_df (DataFrame): 包含日期、投资组合价值和基准的DataFrame
            title (str): 图表标题
            filename (str): 保存文件名
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # 绘制投资组合价值
            plt.plot(
                portfolio_df.index, 
                portfolio_df['portfolio_value'], 
                label='投资组合价值', 
                color='blue', 
                linewidth=2
            )
            
            # 如果有基准数据，绘制基准
            if 'benchmark' in portfolio_df.columns:
                plt.plot(
                    portfolio_df.index, 
                    portfolio_df['benchmark'], 
                    label='基准指数', 
                    color='red', 
                    linestyle='--', 
                    linewidth=1.5
                )
            
            # 如果数据点足够多，添加移动平均线
            if len(portfolio_df) > 20:
                ma20 = portfolio_df['portfolio_value'].rolling(window=20).mean()
                plt.plot(
                    portfolio_df.index, 
                    ma20, 
                    label='20日移动平均', 
                    color='green', 
                    linestyle=':', 
                    linewidth=1.5
                )
            
            # 设置图表格式
            plt.title(title, fontsize=15)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('价值', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # 格式化x轴日期
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
            
            # 添加初始值和最终值标记
            plt.scatter(
                portfolio_df.index[0], 
                portfolio_df['portfolio_value'].iloc[0], 
                color='blue', 
                s=50, 
                zorder=5
            )
            plt.scatter(
                portfolio_df.index[-1], 
                portfolio_df['portfolio_value'].iloc[-1], 
                color='blue', 
                s=50, 
                zorder=5
            )
            
            # 保存图表
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"投资组合表现图表已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"绘制投资组合表现图表时出错: {str(e)}")
            return None
    
    def plot_return_distribution(self, returns_series, title="收益分布", filename="return_distribution.png"):
        """
        绘制收益分布图
        
        参数:
            returns_series (Series): 日收益率序列
            title (str): 图表标题
            filename (str): 保存文件名
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # 绘制收益分布直方图
            sns.histplot(
                returns_series * 100, 
                kde=True, 
                color='blue', 
                alpha=0.6, 
                bins=50, 
                ax=ax1
            )
            
            # 添加统计信息
            mean_return = returns_series.mean() * 100
            median_return = returns_series.median() * 100
            std_return = returns_series.std() * 100
            
            # 在图表上添加统计信息
            stats_text = (
                f"均值: {mean_return:.2f}%\n"
                f"中位数: {median_return:.2f}%\n"
                f"标准差: {std_return:.2f}%\n"
                f"偏度: {returns_series.skew():.2f}\n"
                f"峰度: {returns_series.kurtosis():.2f}"
            )
            
            ax1.text(
                0.95, 0.95, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # 添加均值和中位数线
            ax1.axvline(mean_return, color='red', linestyle='--', linewidth=1.5, label=f'均值: {mean_return:.2f}%')
            ax1.axvline(median_return, color='green', linestyle='-.', linewidth=1.5, label=f'中位数: {median_return:.2f}%')
            ax1.axvline(0, color='black', linestyle='-', linewidth=1.0, label='零收益线')
            
            # 设置第一个子图的格式
            ax1.set_title(title, fontsize=15)
            ax1.set_xlabel('日收益率 (%)', fontsize=12)
            ax1.set_ylabel('频率', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 绘制累积收益图
            cumulative_returns = (1 + returns_series).cumprod() - 1
            cumulative_returns.plot(
                ax=ax2, 
                color='blue', 
                linewidth=2, 
                label='累积收益'
            )
            
            # 设置第二个子图的格式
            ax2.set_title('累积收益', fontsize=15)
            ax2.set_xlabel('日期', fontsize=12)
            ax2.set_ylabel('累积收益 (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
            
            # 保存图表
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"收益分布图表已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"绘制收益分布图表时出错: {str(e)}")
            return None
    
    def plot_drawdown(self, drawdown_series, title="回撤分析", filename="drawdown_analysis.png"):
        """
        绘制回撤分析图
        
        参数:
            drawdown_series (Series): 回撤序列
            title (str): 图表标题
            filename (str): 保存文件名
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # 绘制回撤
            plt.plot(
                drawdown_series.index, 
                drawdown_series * 100, 
                color='red', 
                linewidth=1.5
            )
            
            # 填充回撤区域
            plt.fill_between(
                drawdown_series.index, 
                0, 
                drawdown_series * 100, 
                color='red', 
                alpha=0.3
            )
            
            # 标记最大回撤
            max_drawdown_idx = drawdown_series.idxmin()
            max_drawdown = drawdown_series.min() * 100
            
            plt.scatter(
                max_drawdown_idx, 
                max_drawdown, 
                color='darkred', 
                s=50, 
                zorder=5
            )
            
            plt.annotate(
                f'最大回撤: {max_drawdown:.2f}%',
                xy=(max_drawdown_idx, max_drawdown),
                xytext=(max_drawdown_idx, max_drawdown / 2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center',
                va='bottom',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # 设置图表格式
            plt.title(title, fontsize=15)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('回撤 (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # 格式化y轴为百分比
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
            
            # 格式化x轴日期
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
            
            # 保存图表
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"回撤分析图表已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"绘制回撤分析图表时出错: {str(e)}")
            return None 
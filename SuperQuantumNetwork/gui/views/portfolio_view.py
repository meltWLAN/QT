#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 投资组合视图
展示账户资产和投资组合分析
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.QtGui import QFont, QColor, QBrush, QPainter, QPen
import pyqtgraph as pg
import numpy as np
import pandas as pd


class AccountSummaryWidget(QWidget):
    """账户概览组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建分组框
        account_group = QGroupBox("账户信息")
        account_layout = QFormLayout(account_group)
        
        # 添加账户信息
        self.total_asset_label = QLabel("¥1,000,000.00")
        self.total_asset_label.setFont(QFont("Microsoft YaHei UI", 14, QFont.Bold))
        account_layout.addRow("总资产:", self.total_asset_label)
        
        self.available_cash_label = QLabel("¥500,000.00")
        account_layout.addRow("可用资金:", self.available_cash_label)
        
        self.market_value_label = QLabel("¥500,000.00")
        account_layout.addRow("持仓市值:", self.market_value_label)
        
        self.profit_loss_label = QLabel("+¥50,000.00 (+5.0%)")
        self.profit_loss_label.setStyleSheet("color: red; font-weight: bold;")
        account_layout.addRow("当日盈亏:", self.profit_loss_label)
        
        self.total_profit_loss_label = QLabel("+¥100,000.00 (+10.0%)")
        self.total_profit_loss_label.setStyleSheet("color: red; font-weight: bold;")
        account_layout.addRow("总盈亏:", self.total_profit_loss_label)
        
        # 添加账户分组到主布局
        main_layout.addWidget(account_group)
        
        # 创建风险分组框
        risk_group = QGroupBox("风险控制")
        risk_layout = QFormLayout(risk_group)
        
        # 添加风险信息
        self.max_drawdown_label = QLabel("5.2%")
        risk_layout.addRow("最大回撤:", self.max_drawdown_label)
        
        self.sharpe_label = QLabel("2.35")
        risk_layout.addRow("夏普比率:", self.sharpe_label)
        
        self.volatility_label = QLabel("15.8%")
        risk_layout.addRow("波动率:", self.volatility_label)
        
        self.var_label = QLabel("¥25,000.00")
        risk_layout.addRow("VaR(95%):", self.var_label)
        
        # 添加风险分组到主布局
        main_layout.addWidget(risk_group)
        
        # 添加弹簧，使组件靠上对齐
        main_layout.addStretch(1)
    
    def update_account_data(self, account_data):
        """更新账户数据"""
        # 更新账户信息
        total_asset = account_data.get('total_asset', 0.0)
        self.total_asset_label.setText(f"¥{total_asset:,.2f}")
        
        available_cash = account_data.get('available_cash', 0.0)
        self.available_cash_label.setText(f"¥{available_cash:,.2f}")
        
        market_value = account_data.get('market_value', 0.0)
        self.market_value_label.setText(f"¥{market_value:,.2f}")
        
        daily_profit = account_data.get('daily_profit', 0.0)
        daily_profit_pct = account_data.get('daily_profit_pct', 0.0) * 100
        self.profit_loss_label.setText(f"{'+' if daily_profit >= 0 else ''}¥{daily_profit:,.2f} ({'+' if daily_profit_pct >= 0 else ''}{daily_profit_pct:.2f}%)")
        self.profit_loss_label.setStyleSheet("color: red; font-weight: bold;" if daily_profit >= 0 else "color: green; font-weight: bold;")
        
        total_profit = account_data.get('total_profit', 0.0)
        total_profit_pct = account_data.get('total_profit_pct', 0.0) * 100
        self.total_profit_loss_label.setText(f"{'+' if total_profit >= 0 else ''}¥{total_profit:,.2f} ({'+' if total_profit_pct >= 0 else ''}{total_profit_pct:.2f}%)")
        self.total_profit_loss_label.setStyleSheet("color: red; font-weight: bold;" if total_profit >= 0 else "color: green; font-weight: bold;")
        
        # 更新风险信息
        max_drawdown = account_data.get('max_drawdown', 0.0) * 100
        self.max_drawdown_label.setText(f"{max_drawdown:.2f}%")
        
        sharpe = account_data.get('sharpe', 0.0)
        self.sharpe_label.setText(f"{sharpe:.2f}")
        
        volatility = account_data.get('volatility', 0.0) * 100
        self.volatility_label.setText(f"{volatility:.2f}%")
        
        var = account_data.get('var', 0.0)
        self.var_label.setText(f"¥{var:,.2f}")


class AssetAllocationWidget(QWidget):
    """资产配置组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建示例数据
        self.allocation_data = [
            {'name': '金融', 'value': 0.25, 'color': (255, 0, 0)},
            {'name': '科技', 'value': 0.30, 'color': (0, 255, 0)},
            {'name': '医药', 'value': 0.15, 'color': (0, 0, 255)},
            {'name': '消费', 'value': 0.20, 'color': (255, 255, 0)},
            {'name': '其他', 'value': 0.10, 'color': (128, 128, 128)}
        ]
        
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建饼图容器
        self.plot_widget = pg.PlotWidget()
        main_layout.addWidget(self.plot_widget)
        
        # 设置背景为黑色
        self.plot_widget.setBackground('k')
        
        # 移除轴
        self.plot_widget.getPlotItem().hideAxis('left')
        self.plot_widget.getPlotItem().hideAxis('bottom')
        
        # 绘制饼图
        self._draw_pie_chart()
        
        # 创建表格
        self.allocation_table = QTableWidget()
        self.allocation_table.setColumnCount(3)
        self.allocation_table.setHorizontalHeaderLabels(["板块", "配置比例", "市值"])
        self.allocation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.allocation_table.verticalHeader().setVisible(False)
        self.allocation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # 填充表格
        self._fill_allocation_table()
        
        # 添加表格到主布局
        main_layout.addWidget(self.allocation_table)
    
    def _draw_pie_chart(self):
        """绘制饼图"""
        # 清空图表
        self.plot_widget.clear()
        
        # 创建饼图项目
        pie = pg.PlotDataItem()
        self.plot_widget.addItem(pie)
        
        # 计算总和
        total = sum(item['value'] for item in self.allocation_data)
        
        # 起始角度
        start_angle = 0
        
        # 绘制扇形
        for item in self.allocation_data:
            # 计算角度
            angle = item['value'] / total * 360
            
            # 创建扇形
            sector = pg.QtGui.QGraphicsEllipseItem(-100, -100, 200, 200)
            sector.setStartAngle(int(start_angle * 16))  # Qt中的角度是1/16度
            sector.setSpanAngle(int(angle * 16))
            
            # 设置颜色
            color = QColor(*item['color'])
            sector.setBrush(QBrush(color))
            sector.setPen(QPen(Qt.black, 1))
            
            # 添加到图表
            self.plot_widget.addItem(sector)
            
            # 添加标签
            # 计算标签位置
            label_angle = (start_angle + angle / 2) * np.pi / 180
            label_x = 80 * np.cos(label_angle)
            label_y = 80 * np.sin(label_angle)
            
            # 创建标签
            label = pg.TextItem(text=f"{item['name']}\n{item['value']/total*100:.1f}%", color='w')
            label.setPos(label_x, label_y)
            self.plot_widget.addItem(label)
            
            # 更新起始角度
            start_angle += angle
    
    def _fill_allocation_table(self):
        """填充资产配置表格"""
        # 清空表格
        self.allocation_table.setRowCount(0)
        
        # 检查是否有数据
        if not self.allocation_data:
            return
        
        # 计算总和
        total = sum(item['value'] for item in self.allocation_data)
        
        # 设置行数
        self.allocation_table.setRowCount(len(self.allocation_data))
        
        # 填充数据
        for row, item in enumerate(self.allocation_data):
            # 板块
            name_item = QTableWidgetItem(item['name'])
            self.allocation_table.setItem(row, 0, name_item)
            
            # 配置比例
            ratio = item['value'] / total
            ratio_item = QTableWidgetItem(f"{ratio*100:.2f}%")
            ratio_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.allocation_table.setItem(row, 1, ratio_item)
            
            # 市值
            value = item['value'] * 500000  # 假设总市值500000
            value_item = QTableWidgetItem(f"¥{value:,.2f}")
            value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.allocation_table.setItem(row, 2, value_item)
    
    def update_allocation_data(self, allocation_data):
        """更新资产配置数据"""
        self.allocation_data = allocation_data
        
        # 重新绘制饼图
        self._draw_pie_chart()
        
        # 重新填充表格
        self._fill_allocation_table()


class PortfolioPerformanceWidget(QWidget):
    """投资组合绩效组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建示例数据
        days = 100
        self.dates = pd.date_range(end=pd.Timestamp.now(), periods=days).tolist()
        self.portfolio_values = [1000000 * (1 + 0.001 * i + 0.002 * np.sin(i/10)) for i in range(days)]
        self.benchmark_values = [1000000 * (1 + 0.0008 * i) for i in range(days)]
        
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建图表容器
        self.plot_widget = pg.PlotWidget()
        main_layout.addWidget(self.plot_widget)
        
        # 设置背景为黑色
        self.plot_widget.setBackground('k')
        
        # 显示网格
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # 添加图例
        self.plot_widget.addLegend()
        
        # 绘制绩效曲线
        self._draw_performance_curves()
        
        # 创建绩效指标分组框
        metrics_group = QGroupBox("绩效指标")
        metrics_layout = QFormLayout(metrics_group)
        
        # 添加绩效指标
        self.annual_return_label = QLabel("15.8%")
        metrics_layout.addRow("年化收益率:", self.annual_return_label)
        
        self.alpha_label = QLabel("5.2%")
        metrics_layout.addRow("阿尔法:", self.alpha_label)
        
        self.beta_label = QLabel("0.85")
        metrics_layout.addRow("贝塔:", self.beta_label)
        
        self.sortino_label = QLabel("1.95")
        metrics_layout.addRow("索提诺比率:", self.sortino_label)
        
        self.win_rate_label = QLabel("65.2%")
        metrics_layout.addRow("胜率:", self.win_rate_label)
        
        self.profit_loss_ratio_label = QLabel("2.5")
        metrics_layout.addRow("盈亏比:", self.profit_loss_ratio_label)
        
        # 添加绩效指标分组到主布局
        main_layout.addWidget(metrics_group)
    
    def _draw_performance_curves(self):
        """绘制绩效曲线"""
        # 清空图表
        self.plot_widget.clear()
        
        # 准备数据
        x = range(len(self.dates))
        portfolio_y = self.portfolio_values
        benchmark_y = self.benchmark_values
        
        # 绘制曲线
        portfolio_curve = self.plot_widget.plot(x, portfolio_y, pen=pg.mkPen('r', width=2), name="投资组合")
        benchmark_curve = self.plot_widget.plot(x, benchmark_y, pen=pg.mkPen('b', width=2), name="基准")
    
    def update_performance_data(self, performance_data):
        """更新绩效数据"""
        # 更新绩效指标
        annual_return = performance_data.get('annual_return', 0.0) * 100
        self.annual_return_label.setText(f"{annual_return:.2f}%")
        
        alpha = performance_data.get('alpha', 0.0) * 100
        self.alpha_label.setText(f"{alpha:.2f}%")
        
        beta = performance_data.get('beta', 0.0)
        self.beta_label.setText(f"{beta:.2f}")
        
        sortino = performance_data.get('sortino', 0.0)
        self.sortino_label.setText(f"{sortino:.2f}")
        
        win_rate = performance_data.get('win_rate', 0.0) * 100
        self.win_rate_label.setText(f"{win_rate:.2f}%")
        
        profit_loss_ratio = performance_data.get('profit_loss_ratio', 0.0)
        self.profit_loss_ratio_label.setText(f"{profit_loss_ratio:.2f}")
        
        # 更新绩效曲线
        if 'dates' in performance_data and 'portfolio_values' in performance_data and 'benchmark_values' in performance_data:
            self.dates = performance_data['dates']
            self.portfolio_values = performance_data['portfolio_values']
            self.benchmark_values = performance_data['benchmark_values']
            
            # 重新绘制曲线
            self._draw_performance_curves()


class PortfolioView(QWidget):
    """投资组合视图"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建选项卡部件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 创建账户概览选项卡
        self.account_summary = AccountSummaryWidget()
        self.tab_widget.addTab(self.account_summary, "账户概览")
        
        # 创建资产配置选项卡
        self.asset_allocation = AssetAllocationWidget()
        self.tab_widget.addTab(self.asset_allocation, "资产配置")
        
        # 创建绩效分析选项卡
        self.portfolio_performance = PortfolioPerformanceWidget()
        self.tab_widget.addTab(self.portfolio_performance, "绩效分析")
    
    def initialize_with_data(self, data):
        """使用数据初始化视图"""
        # 更新账户信息
        account_data = data.get("account_data", {})
        self.account_summary.update_account_data(account_data)
        
        # 更新资产配置
        allocation_data = data.get("allocation_data", [])
        if allocation_data:
            self.asset_allocation.update_allocation_data(allocation_data)
        
        # 更新绩效数据
        performance_data = data.get("performance_data", {})
        self.portfolio_performance.update_performance_data(performance_data) 
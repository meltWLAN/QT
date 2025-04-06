#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 分析视图
高级分析功能，包括策略分析、风险分析等
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QSplitter, QGridLayout, QDateEdit
)
from PyQt5.QtCore import Qt, pyqtSlot, QSize, QDate
from PyQt5.QtGui import QFont, QColor, QBrush, QPainter, QPen
import pyqtgraph as pg
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class StrategyAnalysisWidget(QWidget):
    """策略分析组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建示例数据
        self._create_sample_data()
        
        # 设置UI
        self._setup_ui()
    
    def _create_sample_data(self):
        """创建示例数据"""
        # 策略回测数据
        days = 200
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days).tolist()
        
        # 创建几个策略收益曲线
        strategy1 = [100 * (1 + 0.001 * i + 0.003 * np.sin(i/10)) for i in range(days)]
        strategy2 = [100 * (1 + 0.0015 * i + 0.002 * np.cos(i/12)) for i in range(days)]
        strategy3 = [100 * (1 + 0.0012 * i - 0.001 * np.sin(i/15)) for i in range(days)]
        benchmark = [100 * (1 + 0.0008 * i) for i in range(days)]
        
        self.strategy_data = {
            'dates': dates,
            'strategies': {
                '量子动量策略': strategy1,
                '分形套利策略': strategy2,
                '波动跟踪策略': strategy3,
                '基准': benchmark
            }
        }
        
        # 策略统计数据
        self.strategy_stats = [
            {
                'name': '量子动量策略',
                'annual_return': 0.268,
                'sharpe': 2.35,
                'max_drawdown': 0.12,
                'volatility': 0.15,
                'win_rate': 0.68,
                'avg_return': 0.021
            },
            {
                'name': '分形套利策略',
                'annual_return': 0.312,
                'sharpe': 2.56,
                'max_drawdown': 0.15,
                'volatility': 0.18,
                'win_rate': 0.72,
                'avg_return': 0.025
            },
            {
                'name': '波动跟踪策略',
                'annual_return': 0.245,
                'sharpe': 2.18,
                'max_drawdown': 0.10,
                'volatility': 0.13,
                'win_rate': 0.65,
                'avg_return': 0.019
            }
        ]
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建控制区域
        control_layout = QHBoxLayout()
        
        # 策略选择
        control_layout.addWidget(QLabel("策略选择:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(list(self.strategy_data['strategies'].keys()))
        control_layout.addWidget(self.strategy_combo)
        
        # 时间范围选择
        control_layout.addWidget(QLabel("时间范围:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addMonths(-6))
        control_layout.addWidget(self.start_date)
        
        control_layout.addWidget(QLabel("至"))
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        control_layout.addWidget(self.end_date)
        
        # 分析按钮
        self.analyze_button = QPushButton("分析")
        control_layout.addWidget(self.analyze_button)
        
        # 添加控制区域到主布局
        main_layout.addLayout(control_layout)
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, 1)
        
        # 创建图表区域
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建图表容器
        self.plot_widget = pg.PlotWidget()
        chart_layout.addWidget(self.plot_widget)
        
        # 设置背景为黑色
        self.plot_widget.setBackground('k')
        
        # 显示网格
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # 添加图例
        self.plot_widget.addLegend()
        
        # 绘制策略曲线
        self._draw_strategy_curves()
        
        # 添加图表区域到分割器
        splitter.addWidget(chart_widget)
        
        # 创建统计表格区域
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        # 创建统计表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(7)
        self.stats_table.setHorizontalHeaderLabels([
            "策略名称", "年化收益", "夏普比率", "最大回撤", "波动率", "胜率", "平均收益"
        ])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # 填充统计表格
        self._fill_stats_table()
        
        # 添加统计表格到布局
        stats_layout.addWidget(self.stats_table)
        
        # 添加统计区域到分割器
        splitter.addWidget(stats_widget)
        
        # 设置分割器比例
        splitter.setSizes([600, 200])
        
        # 连接信号和槽
        self.strategy_combo.currentIndexChanged.connect(self._strategy_changed)
        self.analyze_button.clicked.connect(self._analyze_strategy)
    
    def _draw_strategy_curves(self):
        """绘制策略曲线"""
        # 清空图表
        self.plot_widget.clear()
        
        # 获取当前选择的策略
        current_strategy = self.strategy_combo.currentText()
        
        # 准备数据
        x = range(len(self.strategy_data['dates']))
        
        # 获取要显示的数据
        selected_strategies = [current_strategy]
        if current_strategy != '基准':
            selected_strategies.append('基准')
        
        # 颜色映射
        colors = {
            '量子动量策略': 'r',
            '分形套利策略': 'g',
            '波动跟踪策略': 'b',
            '基准': 'w'
        }
        
        # 绘制曲线
        for strategy in selected_strategies:
            y = self.strategy_data['strategies'][strategy]
            self.plot_widget.plot(
                x, y,
                pen=pg.mkPen(colors.get(strategy, 'w'), width=2),
                name=strategy
            )
    
    def _fill_stats_table(self):
        """填充统计表格"""
        # 清空表格
        self.stats_table.setRowCount(0)
        
        # 设置行数
        self.stats_table.setRowCount(len(self.strategy_stats))
        
        # 填充数据
        for row, stats in enumerate(self.strategy_stats):
            # 策略名称
            name_item = QTableWidgetItem(stats['name'])
            self.stats_table.setItem(row, 0, name_item)
            
            # 年化收益
            return_item = QTableWidgetItem(f"{stats['annual_return']*100:.2f}%")
            return_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(row, 1, return_item)
            
            # 夏普比率
            sharpe_item = QTableWidgetItem(f"{stats['sharpe']:.2f}")
            sharpe_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(row, 2, sharpe_item)
            
            # 最大回撤
            drawdown_item = QTableWidgetItem(f"{stats['max_drawdown']*100:.2f}%")
            drawdown_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(row, 3, drawdown_item)
            
            # 波动率
            volatility_item = QTableWidgetItem(f"{stats['volatility']*100:.2f}%")
            volatility_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(row, 4, volatility_item)
            
            # 胜率
            win_rate_item = QTableWidgetItem(f"{stats['win_rate']*100:.2f}%")
            win_rate_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(row, 5, win_rate_item)
            
            # 平均收益
            avg_return_item = QTableWidgetItem(f"{stats['avg_return']*100:.2f}%")
            avg_return_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(row, 6, avg_return_item)
    
    def _strategy_changed(self, index):
        """策略改变处理"""
        self._draw_strategy_curves()
    
    def _analyze_strategy(self):
        """分析策略"""
        # 获取选定的日期范围
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        
        # 更新图表和统计信息
        # TODO: 实现实际的日期过滤和数据更新逻辑
        self._draw_strategy_curves()
    
    def update_strategy_data(self, strategy_data, strategy_stats):
        """更新策略数据"""
        if strategy_data:
            self.strategy_data = strategy_data
            
            # 更新策略选择下拉框
            self.strategy_combo.clear()
            self.strategy_combo.addItems(list(strategy_data['strategies'].keys()))
            
            # 重新绘制曲线
            self._draw_strategy_curves()
        
        if strategy_stats:
            self.strategy_stats = strategy_stats
            
            # 重新填充表格
            self._fill_stats_table()


class RiskAnalysisWidget(QWidget):
    """风险分析组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建示例数据
        self._create_sample_data()
        
        # 设置UI
        self._setup_ui()
    
    def _create_sample_data(self):
        """创建示例数据"""
        # 相关性矩阵数据
        stocks = ["工商银行", "茅台", "腾讯", "阿里巴巴", "平安保险", "中国石油", "中国移动", "恒瑞医药", "格力电器", "万科A"]
        n = len(stocks)
        
        # 创建相关性矩阵
        correlation_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # 生成一个-0.5到0.9之间的随机相关系数
                    correlation_matrix[i, j] = np.random.uniform(-0.5, 0.9)
                    correlation_matrix[j, i] = correlation_matrix[i, j]  # 确保对称
        
        self.correlation_data = {
            'stocks': stocks,
            'matrix': correlation_matrix
        }
        
        # 风险分解数据
        self.risk_decomposition = [
            {'name': '市场风险', 'value': 0.45, 'color': (255, 0, 0)},
            {'name': '特异风险', 'value': 0.25, 'color': (0, 255, 0)},
            {'name': '行业风险', 'value': 0.15, 'color': (0, 0, 255)},
            {'name': '风格风险', 'value': 0.10, 'color': (255, 255, 0)},
            {'name': '其他风险', 'value': 0.05, 'color': (128, 128, 128)}
        ]
        
        # 风险指标数据
        self.risk_metrics = {
            'var': 0.025,  # 95% VaR
            'cvar': 0.035,  # 95% CVaR
            'volatility': 0.15,  # 年化波动率
            'max_drawdown': 0.12,  # 最大回撤
            'downside_risk': 0.08,  # 下行风险
            'beta': 0.85,  # Beta
            'tracking_error': 0.05  # 跟踪误差
        }
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建选项卡部件
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # 创建相关性分析选项卡
        correlation_widget = QWidget()
        correlation_layout = QVBoxLayout(correlation_widget)
        
        # 创建热力图容器
        self.heatmap_widget = pg.PlotWidget()
        correlation_layout.addWidget(self.heatmap_widget)
        
        # 设置背景为黑色
        self.heatmap_widget.setBackground('k')
        
        # 绘制相关性热力图
        self._draw_correlation_heatmap()
        
        # 添加相关性分析选项卡
        tab_widget.addTab(correlation_widget, "相关性分析")
        
        # 创建风险分解选项卡
        decomposition_widget = QWidget()
        decomposition_layout = QVBoxLayout(decomposition_widget)
        
        # 创建饼图容器
        self.pie_widget = pg.PlotWidget()
        decomposition_layout.addWidget(self.pie_widget)
        
        # 设置背景为黑色
        self.pie_widget.setBackground('k')
        
        # 移除轴
        self.pie_widget.getPlotItem().hideAxis('left')
        self.pie_widget.getPlotItem().hideAxis('bottom')
        
        # 绘制风险分解饼图
        self._draw_risk_decomposition_pie()
        
        # 添加风险分解选项卡
        tab_widget.addTab(decomposition_widget, "风险分解")
        
        # 创建风险指标选项卡
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        # 创建风险指标分组框
        metrics_group = QGroupBox("量化风险指标")
        metrics_form = QFormLayout(metrics_group)
        
        # 添加风险指标
        self.var_label = QLabel(f"{self.risk_metrics['var']*100:.2f}%")
        metrics_form.addRow("Value at Risk (95%):", self.var_label)
        
        self.cvar_label = QLabel(f"{self.risk_metrics['cvar']*100:.2f}%")
        metrics_form.addRow("Conditional VaR:", self.cvar_label)
        
        self.volatility_label = QLabel(f"{self.risk_metrics['volatility']*100:.2f}%")
        metrics_form.addRow("年化波动率:", self.volatility_label)
        
        self.max_drawdown_label = QLabel(f"{self.risk_metrics['max_drawdown']*100:.2f}%")
        metrics_form.addRow("最大回撤:", self.max_drawdown_label)
        
        self.downside_risk_label = QLabel(f"{self.risk_metrics['downside_risk']*100:.2f}%")
        metrics_form.addRow("下行风险:", self.downside_risk_label)
        
        self.beta_label = QLabel(f"{self.risk_metrics['beta']:.2f}")
        metrics_form.addRow("Beta:", self.beta_label)
        
        self.tracking_error_label = QLabel(f"{self.risk_metrics['tracking_error']*100:.2f}%")
        metrics_form.addRow("跟踪误差:", self.tracking_error_label)
        
        # 添加风险指标分组到布局
        metrics_layout.addWidget(metrics_group)
        
        # 添加指标说明
        explanation_text = """
        <b>风险指标说明:</b><br>
        <b>Value at Risk (VaR):</b> 在给定置信水平下，预期的最大损失<br>
        <b>Conditional VaR (CVaR):</b> 超过VaR的平均损失<br>
        <b>年化波动率:</b> 收益率的标准差，年化<br>
        <b>最大回撤:</b> 从历史最高点到最低点的最大下跌幅度<br>
        <b>下行风险:</b> 只考虑负收益波动的风险度量<br>
        <b>Beta:</b> 相对于市场的风险度量<br>
        <b>跟踪误差:</b> 相对于基准的偏离程度
        """
        explanation_label = QLabel(explanation_text)
        explanation_label.setWordWrap(True)
        metrics_layout.addWidget(explanation_label)
        
        # 添加空白空间
        metrics_layout.addStretch(1)
        
        # 添加风险指标选项卡
        tab_widget.addTab(metrics_widget, "风险指标")
    
    def _draw_correlation_heatmap(self):
        """绘制相关性热力图"""
        # 清空图表
        self.heatmap_widget.clear()
        
        # 获取数据
        stocks = self.correlation_data['stocks']
        matrix = self.correlation_data['matrix']
        n = len(stocks)
        
        # 创建图像项
        img = pg.ImageItem()
        self.heatmap_widget.addItem(img)
        
        # 设置图像数据
        img.setImage(matrix)
        
        # 设置位置和缩放
        img.scale(1, 1)
        img.translate(-0.5, -0.5)
        
        # 添加颜色条
        colorbar = pg.ColorBarItem(
            values=(-1, 1),
            colorMap=pg.ColorMap(
                pos=np.linspace(0.0, 1.0, 3),
                color=[(0, 0, 255, 255), (255, 255, 255, 255), (255, 0, 0, 255)]
            )
        )
        colorbar.setImageItem(img)
        self.heatmap_widget.addItem(colorbar)
        
        # 添加坐标轴
        ax = self.heatmap_widget.getAxis('bottom')
        ax.setTicks([[(i, stocks[i]) for i in range(n)]])
        
        ay = self.heatmap_widget.getAxis('left')
        ay.setTicks([[(i, stocks[i]) for i in range(n)]])
    
    def _draw_risk_decomposition_pie(self):
        """绘制风险分解饼图"""
        # 清空图表
        self.pie_widget.clear()
        
        # 创建饼图项目
        pie = pg.PlotDataItem()
        self.pie_widget.addItem(pie)
        
        # 计算总和
        total = sum(item['value'] for item in self.risk_decomposition)
        
        # 起始角度
        start_angle = 0
        
        # 绘制扇形
        for item in self.risk_decomposition:
            # 计算角度
            angle = item['value'] / total * 360
            
            # 创建扇形
            sector = pg.QtGui.QGraphicsEllipseItem(-100, -100, 200, 200)
            sector.setStartAngle(start_angle * 16)  # Qt中的角度是1/16度
            sector.setSpanAngle(angle * 16)
            
            # 设置颜色
            color = QColor(*item['color'])
            sector.setBrush(QBrush(color))
            sector.setPen(QPen(Qt.black, 1))
            
            # 添加到图表
            self.pie_widget.addItem(sector)
            
            # 添加标签
            # 计算标签位置
            label_angle = (start_angle + angle / 2) * np.pi / 180
            label_x = 80 * np.cos(label_angle)
            label_y = 80 * np.sin(label_angle)
            
            # 创建标签
            label = pg.TextItem(text=f"{item['name']}\n{item['value']/total*100:.1f}%", color='w')
            label.setPos(label_x, label_y)
            self.pie_widget.addItem(label)
            
            # 更新起始角度
            start_angle += angle
    
    def update_risk_data(self, correlation_data=None, risk_decomposition=None, risk_metrics=None):
        """更新风险数据"""
        if correlation_data:
            self.correlation_data = correlation_data
            self._draw_correlation_heatmap()
        
        if risk_decomposition:
            self.risk_decomposition = risk_decomposition
            self._draw_risk_decomposition_pie()
        
        if risk_metrics:
            self.risk_metrics = risk_metrics
            
            # 更新风险指标标签
            self.var_label.setText(f"{self.risk_metrics['var']*100:.2f}%")
            self.cvar_label.setText(f"{self.risk_metrics['cvar']*100:.2f}%")
            self.volatility_label.setText(f"{self.risk_metrics['volatility']*100:.2f}%")
            self.max_drawdown_label.setText(f"{self.risk_metrics['max_drawdown']*100:.2f}%")
            self.downside_risk_label.setText(f"{self.risk_metrics['downside_risk']*100:.2f}%")
            self.beta_label.setText(f"{self.risk_metrics['beta']:.2f}")
            self.tracking_error_label.setText(f"{self.risk_metrics['tracking_error']*100:.2f}%")


class AnalysisView(QWidget):
    """分析视图"""
    
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
        
        # 创建策略分析选项卡
        self.strategy_analysis = StrategyAnalysisWidget()
        self.tab_widget.addTab(self.strategy_analysis, "策略分析")
        
        # 创建风险分析选项卡
        self.risk_analysis = RiskAnalysisWidget()
        self.tab_widget.addTab(self.risk_analysis, "风险分析")
    
    def initialize_with_data(self, data):
        """使用数据初始化视图"""
        # 更新策略分析
        strategy_data = data.get("strategy_data")
        strategy_stats = data.get("strategy_stats")
        if strategy_data or strategy_stats:
            self.strategy_analysis.update_strategy_data(strategy_data, strategy_stats)
        
        # 更新风险分析
        correlation_data = data.get("correlation_data")
        risk_decomposition = data.get("risk_decomposition")
        risk_metrics = data.get("risk_metrics")
        if correlation_data or risk_decomposition or risk_metrics:
            self.risk_analysis.update_risk_data(correlation_data, risk_decomposition, risk_metrics) 
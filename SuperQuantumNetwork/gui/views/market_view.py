#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 市场视图
展示市场行情、指数和热门股票
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QComboBox, QPushButton, QHeaderView, QTabWidget, QSplitter,
    QGridLayout, QFrame
)
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QBrush, QIcon
import pyqtgraph as pg
import pandas as pd
import numpy as np


class MarketIndexWidget(QWidget):
    """市场指数小部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建布局
        layout = QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 添加指数面板
        self._create_index_panel("上证指数", "000001.SH", 0, 0)
        self._create_index_panel("深证成指", "399001.SZ", 0, 1)
        self._create_index_panel("创业板指", "399006.SZ", 1, 0)
        self._create_index_panel("科创50", "000688.SH", 1, 1)
    
    def _create_index_panel(self, name, code, row, col):
        """创建单个指数面板"""
        # 创建框架
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setLineWidth(1)
        
        # 创建框架布局
        frame_layout = QVBoxLayout(frame)
        
        # 指数名称
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
        
        # 指数代码
        code_label = QLabel(code)
        code_label.setAlignment(Qt.AlignCenter)
        code_label.setFont(QFont("Microsoft YaHei UI", 9))
        
        # 指数价格
        self.price_label = QLabel("0.00")
        self.price_label.setAlignment(Qt.AlignCenter)
        self.price_label.setFont(QFont("Microsoft YaHei UI", 18, QFont.Bold))
        
        # 涨跌幅
        self.change_label = QLabel("0.00%")
        self.change_label.setAlignment(Qt.AlignCenter)
        self.change_label.setFont(QFont("Microsoft YaHei UI", 14))
        
        # 添加组件到布局
        frame_layout.addWidget(name_label)
        frame_layout.addWidget(code_label)
        frame_layout.addWidget(self.price_label)
        frame_layout.addWidget(self.change_label)
        
        # 添加框架到主布局
        self.layout().addWidget(frame, row, col)
    
    def update_index_data(self, data):
        """更新指数数据"""
        # 实现指数数据更新逻辑
        pass


class MarketHeatMapWidget(QWidget):
    """市场热力图小部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建控制区域
        control_layout = QHBoxLayout()
        
        # 热力图类型选择
        self.heat_type_combo = QComboBox()
        self.heat_type_combo.addItems(["行业板块", "概念板块", "涨跌幅", "成交量"])
        control_layout.addWidget(QLabel("热力图类型:"))
        control_layout.addWidget(self.heat_type_combo)
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新")
        control_layout.addWidget(self.refresh_button)
        
        # 添加控制区域到主布局
        layout.addLayout(control_layout)
        
        # 创建热力图区域
        self.heatmap_view = pg.GraphicsLayoutWidget()
        layout.addWidget(self.heatmap_view, 1)  # 1表示拉伸因子
    
    def update_heatmap_data(self, data):
        """更新热力图数据"""
        # 实现热力图数据更新逻辑
        pass


class HotStocksTableWidget(QWidget):
    """热门股票表格小部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建控制区域
        control_layout = QHBoxLayout()
        
        # 股票类型选择
        self.stock_type_combo = QComboBox()
        self.stock_type_combo.addItems(["量子网络推荐", "涨幅榜", "跌幅榜", "成交额榜"])
        control_layout.addWidget(QLabel("股票类型:"))
        control_layout.addWidget(self.stock_type_combo)
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新")
        control_layout.addWidget(self.refresh_button)
        
        # 添加控制区域到主布局
        layout.addLayout(control_layout)
        
        # 创建表格
        self.stocks_table = QTableWidget()
        self.stocks_table.setColumnCount(6)
        self.stocks_table.setHorizontalHeaderLabels(["代码", "名称", "最新价", "涨跌幅", "成交量", "推荐度"])
        self.stocks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stocks_table.verticalHeader().setVisible(False)
        self.stocks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stocks_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # 添加表格到主布局
        layout.addWidget(self.stocks_table, 1)  # 1表示拉伸因子
    
    def update_stocks_data(self, data):
        """更新股票数据"""
        # 清空表格
        self.stocks_table.setRowCount(0)
        
        # 检查是否有数据
        if not data or not isinstance(data, list):
            return
        
        # 设置行数
        self.stocks_table.setRowCount(len(data))
        
        # 填充数据
        for row, stock in enumerate(data):
            # 代码
            code_item = QTableWidgetItem(stock.get('code', ''))
            self.stocks_table.setItem(row, 0, code_item)
            
            # 名称
            name_item = QTableWidgetItem(stock.get('name', ''))
            self.stocks_table.setItem(row, 1, name_item)
            
            # 最新价
            price_item = QTableWidgetItem(f"{stock.get('price', 0.0):.2f}")
            price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stocks_table.setItem(row, 2, price_item)
            
            # 涨跌幅
            change = stock.get('change', 0.0)
            change_item = QTableWidgetItem(f"{change:.2f}%")
            change_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            # 设置涨跌幅颜色
            if change > 0:
                change_item.setForeground(QBrush(QColor(255, 0, 0)))
            elif change < 0:
                change_item.setForeground(QBrush(QColor(0, 255, 0)))
                
            self.stocks_table.setItem(row, 3, change_item)
            
            # 成交量
            volume = stock.get('volume', 0)
            volume_text = f"{volume/10000:.2f}万" if volume < 10000000 else f"{volume/100000000:.2f}亿"
            volume_item = QTableWidgetItem(volume_text)
            volume_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stocks_table.setItem(row, 4, volume_item)
            
            # 推荐度
            recommendation = stock.get('recommendation', 0.0)
            recommendation_item = QTableWidgetItem(f"{recommendation:.2f}")
            recommendation_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stocks_table.setItem(row, 5, recommendation_item)


class MarketView(QWidget):
    """市场视图"""
    
    def __init__(self, data_controller, parent=None):
        super().__init__(parent)
        self.data_controller = data_controller
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建市场指数部件
        self.index_widget = MarketIndexWidget()
        main_layout.addWidget(self.index_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, 1)  # 1表示拉伸因子
        
        # 创建热力图部件
        self.heatmap_widget = MarketHeatMapWidget()
        splitter.addWidget(self.heatmap_widget)
        
        # 创建热门股票表格部件
        self.hot_stocks_widget = HotStocksTableWidget()
        splitter.addWidget(self.hot_stocks_widget)
        
        # 设置分割器初始大小
        splitter.setSizes([200, 300])
    
    def initialize_with_data(self, data):
        """使用数据初始化视图"""
        # 更新热门股票
        recommended_stocks = data.get("recommended_stocks", [])
        self.hot_stocks_widget.update_stocks_data(recommended_stocks)
        
        # 更新其他数据
        # self.index_widget.update_index_data(data.get("index_data", {}))
        # self.heatmap_widget.update_heatmap_data(data) 
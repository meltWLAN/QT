#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 图表视图
实现K线图和各种技术指标显示
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, 
    QLabel, QToolBar, QAction, QSplitter, QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.QtGui import QIcon, QFont
import pyqtgraph as pg
import numpy as np
import pandas as pd
from datetime import datetime
import qtawesome as qta


class CandlestickItem(pg.GraphicsObject):
    """K线图项目"""
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.picture = None
        self.generatePicture()
    
    def generatePicture(self):
        """生成K线图"""
        if self.data is None or len(self.data) == 0:
            return
            
        self.picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self.picture)
        painter.setPen(pg.mkPen('w'))
        
        w = 0.4  # 蜡烛宽度
        for i in range(len(self.data)):
            if i >= len(self.data['open']):
                continue
                
            open_val = self.data['open'][i]
            high = self.data['high'][i]
            low = self.data['low'][i]
            close_val = self.data['close'][i]
            
            # 根据收盘价与开盘价的关系设置颜色
            if close_val > open_val:
                color = pg.mkColor('r')  # 红色表示上涨
            elif close_val < open_val:
                color = pg.mkColor('g')  # 绿色表示下跌
            else:
                color = pg.mkColor('w')  # 白色表示平盘
            
            # 画上下影线
            painter.setPen(pg.mkPen(color, width=1))
            painter.drawLine(pg.QtCore.QPointF(i, low), pg.QtCore.QPointF(i, high))
            
            # 画实体
            painter.setPen(pg.mkPen(color, width=1))
            if close_val > open_val:
                painter.setBrush(pg.mkBrush(color))
            else:
                painter.setBrush(pg.mkBrush(None))
                
            rect = pg.QtCore.QRectF(i - w, open_val, w * 2, close_val - open_val)
            painter.drawRect(rect)
            
        painter.end()
    
    def paint(self, painter, option, widget):
        """绘制K线图"""
        if self.picture is not None:
            painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        """返回边界矩形"""
        if self.picture is None:
            return pg.QtCore.QRectF(0, 0, 0, 0)
        return pg.QtCore.QRectF(self.picture.boundingRect())
    
    def update_data(self, data):
        """更新数据"""
        self.data = data
        self.generatePicture()
        self.update()


class VolumeItem(pg.BarGraphItem):
    """成交量图项目"""
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.update_data(data)
    
    def update_data(self, data):
        """更新数据"""
        self.data = data
        
        if self.data is None or len(self.data) == 0:
            return
            
        # 提取数据
        x = np.arange(len(self.data))
        height = self.data['volume']
        
        # 根据收盘价与开盘价的关系设置颜色
        colors = []
        brushes = []
        for i in range(len(self.data)):
            if i >= len(self.data['open']):
                brushes.append((255, 255, 255))  # 白色
                continue
                
            if self.data['close'][i] > self.data['open'][i]:
                brushes.append((255, 0, 0))  # 红色表示上涨
            elif self.data['close'][i] < self.data['open'][i]:
                brushes.append((0, 255, 0))  # 绿色表示下跌
            else:
                brushes.append((255, 255, 255))  # 白色表示平盘
        
        self.setOpts(x=x, height=height, width=0.8, brush=brushes)


class ChartView(QWidget):
    """图表视图"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建默认数据
        self.default_data = {
            'date': np.array([datetime.now().timestamp() - i * 86400 for i in range(100)]),
            'open': np.random.normal(100, 2, 100),
            'high': np.zeros(100),
            'low': np.zeros(100),
            'close': np.zeros(100),
            'volume': np.random.normal(1000000, 200000, 100),
        }
        
        # 初始化其他值
        for i in range(100):
            self.default_data['high'][i] = max(self.default_data['open'][i] + np.random.normal(0, 1), self.default_data['open'][i])
            self.default_data['low'][i] = min(self.default_data['open'][i] - np.random.normal(0, 1), self.default_data['open'][i])
            self.default_data['close'][i] = np.random.normal(self.default_data['open'][i], 0.5)
            
        # 转换为DataFrame
        self.data = pd.DataFrame(self.default_data)
        
        # 设置UI
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建工具栏
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))
        
        # 添加周期选择
        self.period_combo = QComboBox()
        self.period_combo.addItems(["分时", "1分钟", "5分钟", "15分钟", "30分钟", "1小时", "日线", "周线", "月线"])
        self.period_combo.setCurrentIndex(6)  # 默认选择日线
        toolbar.addWidget(QLabel("周期: "))
        toolbar.addWidget(self.period_combo)
        toolbar.addSeparator()
        
        # 添加技术指标选择
        self.indicator_combo = QComboBox()
        self.indicator_combo.addItems(["MA", "MACD", "KDJ", "RSI", "BOLL", "VOL"])
        toolbar.addWidget(QLabel("指标: "))
        toolbar.addWidget(self.indicator_combo)
        
        # 添加刷新按钮
        refresh_action = QAction(qta.icon('fa5s.sync'), "刷新", self)
        toolbar.addAction(refresh_action)
        
        # 添加放大/缩小按钮
        zoom_in_action = QAction(qta.icon('fa5s.search-plus'), "放大", self)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction(qta.icon('fa5s.search-minus'), "缩小", self)
        toolbar.addAction(zoom_out_action)
        
        # 添加十字光标按钮
        crosshair_action = QAction(qta.icon('fa5s.crosshairs'), "十字光标", self)
        crosshair_action.setCheckable(True)
        toolbar.addAction(crosshair_action)
        
        # 添加工具栏到主布局
        main_layout.addWidget(toolbar)
        
        # 创建图表布局
        self.chart_layout = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.chart_layout)
        
        # 设置背景为黑色
        self.chart_layout.setBackground('k')
        
        # 创建K线图
        self.price_plot = self.chart_layout.addPlot(row=0, col=0)
        self.price_plot.setAutoVisible(y=True)
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setLabel('left', 'Price')
        
        # 添加十字光标
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.price_plot.addItem(self.vLine, ignoreBounds=True)
        self.price_plot.addItem(self.hLine, ignoreBounds=True)
        
        # 创建成交量图
        self.volume_plot = self.chart_layout.addPlot(row=1, col=0)
        self.volume_plot.setAutoVisible(y=True)
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setLabel('left', 'Volume')
        
        # 创建指标图
        self.indicator_plot = self.chart_layout.addPlot(row=2, col=0)
        self.indicator_plot.setAutoVisible(y=True)
        self.indicator_plot.showGrid(x=True, y=True, alpha=0.3)
        self.indicator_plot.setLabel('left', 'Indicator')
        
        # 链接X轴
        self.volume_plot.setXLink(self.price_plot)
        self.indicator_plot.setXLink(self.price_plot)
        
        # 设置图表大小比例
        self.chart_layout.ci.layout.setRowStretchFactor(0, 3)
        self.chart_layout.ci.layout.setRowStretchFactor(1, 1)
        self.chart_layout.ci.layout.setRowStretchFactor(2, 1)
        
        # 创建并添加K线图
        self.candlestick_item = CandlestickItem(self.data)
        self.price_plot.addItem(self.candlestick_item)
        
        # 创建并添加成交量图
        self.volume_item = VolumeItem(self.data)
        self.volume_plot.addItem(self.volume_item)
        
        # 连接信号和槽
        self.price_plot.scene().sigMouseMoved.connect(self._mouse_moved)
        refresh_action.triggered.connect(self._refresh_chart)
        crosshair_action.triggered.connect(self._toggle_crosshair)
        self.period_combo.currentIndexChanged.connect(self._period_changed)
        self.indicator_combo.currentIndexChanged.connect(self._indicator_changed)
        
        # 默认隐藏十字光标
        self.vLine.hide()
        self.hLine.hide()
    
    def _mouse_moved(self, pos):
        """鼠标移动处理"""
        if self.price_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.price_plot.vb.mapSceneToView(pos)
            self.vLine.setPos(mouse_point.x())
            self.hLine.setPos(mouse_point.y())
    
    def _toggle_crosshair(self, checked):
        """切换十字光标"""
        if checked:
            self.vLine.show()
            self.hLine.show()
        else:
            self.vLine.hide()
            self.hLine.hide()
    
    def _refresh_chart(self):
        """刷新图表"""
        # TODO: 获取实时数据并更新图表
        pass
    
    def _period_changed(self, index):
        """周期改变处理"""
        # TODO: 根据选择的周期更新数据
        pass
    
    def _indicator_changed(self, index):
        """指标改变处理"""
        # TODO: 根据选择的指标更新数据
        pass
    
    def initialize_with_data(self, data):
        """使用数据初始化视图"""
        # 检查是否有股票数据
        stocks = data.get("stocks", {})
        if stocks:
            # 获取第一只股票的数据
            first_stock = list(stocks.values())[0]
            # 更新图表
            self._update_chart_data(first_stock)
    
    def _update_chart_data(self, stock_data):
        """更新图表数据"""
        # TODO: 将股票数据转换为图表数据格式，并更新图表
        pass 
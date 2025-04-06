#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 交易视图
实现交易下单、持仓管理等功能
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QFormLayout, QRadioButton, QButtonGroup, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.QtGui import QFont, QColor, QBrush, QIcon


class OrderWidget(QWidget):
    """下单组件"""
    
    def __init__(self, trading_controller, parent=None):
        super().__init__(parent)
        self.trading_controller = trading_controller
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # 股票代码和名称组
        code_group = QGroupBox("股票信息")
        code_layout = QFormLayout(code_group)
        
        # 股票代码输入
        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("输入股票代码...")
        code_layout.addRow("股票代码:", self.code_input)
        
        # 股票名称显示
        self.name_label = QLabel("--")
        code_layout.addRow("股票名称:", self.name_label)
        
        # 最新价格显示
        self.price_label = QLabel("0.00")
        self.price_label.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
        code_layout.addRow("最新价格:", self.price_label)
        
        # 添加股票信息组到主布局
        main_layout.addWidget(code_group)
        
        # 创建交易参数组
        trade_group = QGroupBox("交易参数")
        trade_layout = QFormLayout(trade_group)
        
        # 交易方向
        direction_layout = QHBoxLayout()
        self.buy_radio = QRadioButton("买入")
        self.sell_radio = QRadioButton("卖出")
        self.buy_radio.setChecked(True)
        direction_layout.addWidget(self.buy_radio)
        direction_layout.addWidget(self.sell_radio)
        trade_layout.addRow("交易方向:", direction_layout)
        
        # 订单类型
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["限价单", "市价单", "止损单", "止盈单"])
        trade_layout.addRow("订单类型:", self.order_type_combo)
        
        # 价格输入
        self.price_input = QDoubleSpinBox()
        self.price_input.setRange(0, 10000)
        self.price_input.setDecimals(2)
        self.price_input.setSingleStep(0.01)
        trade_layout.addRow("价格:", self.price_input)
        
        # 数量输入
        self.quantity_input = QSpinBox()
        self.quantity_input.setRange(100, 1000000)
        self.quantity_input.setSingleStep(100)
        self.quantity_input.setValue(100)
        trade_layout.addRow("数量:", self.quantity_input)
        
        # 总金额显示
        self.total_amount_label = QLabel("0.00")
        trade_layout.addRow("总金额:", self.total_amount_label)
        
        # 添加交易参数组到主布局
        main_layout.addWidget(trade_group)
        
        # 高级设置组
        advanced_group = QGroupBox("高级设置")
        advanced_layout = QFormLayout(advanced_group)
        
        # 止损价
        self.stop_loss_input = QDoubleSpinBox()
        self.stop_loss_input.setRange(0, 10000)
        self.stop_loss_input.setDecimals(2)
        self.stop_loss_input.setSingleStep(0.01)
        advanced_layout.addRow("止损价:", self.stop_loss_input)
        
        # 止盈价
        self.take_profit_input = QDoubleSpinBox()
        self.take_profit_input.setRange(0, 10000)
        self.take_profit_input.setDecimals(2)
        self.take_profit_input.setSingleStep(0.01)
        advanced_layout.addRow("止盈价:", self.take_profit_input)
        
        # 下单时间有效期
        self.time_in_force_combo = QComboBox()
        self.time_in_force_combo.addItems(["当日有效", "撤单前有效", "指定日期"])
        advanced_layout.addRow("有效期:", self.time_in_force_combo)
        
        # 添加高级设置组到主布局
        main_layout.addWidget(advanced_group)
        
        # 创建下单按钮
        button_layout = QHBoxLayout()
        
        # 查询按钮
        self.query_button = QPushButton("查询行情")
        button_layout.addWidget(self.query_button)
        
        # 下单按钮
        self.submit_button = QPushButton("下单")
        self.submit_button.setFont(QFont("Microsoft YaHei UI", 11, QFont.Bold))
        button_layout.addWidget(self.submit_button)
        
        # 重置按钮
        self.reset_button = QPushButton("重置")
        button_layout.addWidget(self.reset_button)
        
        # 添加按钮布局到主布局
        main_layout.addLayout(button_layout)
        
        # 添加弹簧，使组件靠上对齐
        main_layout.addStretch(1)
        
        # 连接信号和槽
        self.code_input.textChanged.connect(self._on_code_changed)
        self.price_input.valueChanged.connect(self._update_total_amount)
        self.quantity_input.valueChanged.connect(self._update_total_amount)
        self.query_button.clicked.connect(self._on_query_clicked)
        self.submit_button.clicked.connect(self._on_submit_clicked)
        self.reset_button.clicked.connect(self._reset_form)
    
    def _on_code_changed(self, code):
        """股票代码改变时的处理"""
        # TODO: 查询股票名称和价格
        self.name_label.setText("--")
        self.price_label.setText("0.00")
    
    def _update_total_amount(self):
        """更新总金额"""
        price = self.price_input.value()
        quantity = self.quantity_input.value()
        total = price * quantity
        self.total_amount_label.setText(f"{total:.2f}")
    
    def _on_query_clicked(self):
        """查询按钮点击处理"""
        code = self.code_input.text().strip()
        if not code:
            return
        
        # TODO: 查询股票实时数据
        
    def _on_submit_clicked(self):
        """下单按钮点击处理"""
        # TODO: 提交订单
        
    def _reset_form(self):
        """重置表单"""
        self.code_input.clear()
        self.name_label.setText("--")
        self.price_label.setText("0.00")
        self.price_input.setValue(0)
        self.quantity_input.setValue(100)
        self.stop_loss_input.setValue(0)
        self.take_profit_input.setValue(0)
        self.buy_radio.setChecked(True)
        self.total_amount_label.setText("0.00")


class PositionTableWidget(QWidget):
    """持仓表格组件"""
    
    def __init__(self, trading_controller, parent=None):
        super().__init__(parent)
        self.trading_controller = trading_controller
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建控制区域
        control_layout = QHBoxLayout()
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新")
        control_layout.addWidget(self.refresh_button)
        
        # 平仓按钮
        self.close_position_button = QPushButton("平仓")
        control_layout.addWidget(self.close_position_button)
        
        # 一键平仓按钮
        self.close_all_button = QPushButton("一键平仓")
        control_layout.addWidget(self.close_all_button)
        
        # 添加控制区域到主布局
        main_layout.addLayout(control_layout)
        
        # 创建持仓表格
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(7)
        self.position_table.setHorizontalHeaderLabels(["代码", "名称", "持仓量", "可用量", "成本价", "现价", "盈亏"])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.position_table.verticalHeader().setVisible(False)
        self.position_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.position_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # 添加表格到主布局
        main_layout.addWidget(self.position_table)
        
        # 连接信号和槽
        self.refresh_button.clicked.connect(self._refresh_positions)
    
    def _refresh_positions(self):
        """刷新持仓数据"""
        # TODO: 获取并更新持仓数据
        pass
    
    def update_positions(self, positions):
        """更新持仓数据"""
        # 清空表格
        self.position_table.setRowCount(0)
        
        # 检查是否有数据
        if not positions or not isinstance(positions, list):
            return
        
        # 设置行数
        self.position_table.setRowCount(len(positions))
        
        # 填充数据
        for row, position in enumerate(positions):
            # 代码
            code_item = QTableWidgetItem(position.get('code', ''))
            self.position_table.setItem(row, 0, code_item)
            
            # 名称
            name_item = QTableWidgetItem(position.get('name', ''))
            self.position_table.setItem(row, 1, name_item)
            
            # 持仓量
            quantity_item = QTableWidgetItem(str(position.get('quantity', 0)))
            quantity_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.position_table.setItem(row, 2, quantity_item)
            
            # 可用量
            available_item = QTableWidgetItem(str(position.get('available', 0)))
            available_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.position_table.setItem(row, 3, available_item)
            
            # 成本价
            cost_item = QTableWidgetItem(f"{position.get('cost', 0.0):.2f}")
            cost_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.position_table.setItem(row, 4, cost_item)
            
            # 现价
            price_item = QTableWidgetItem(f"{position.get('price', 0.0):.2f}")
            price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.position_table.setItem(row, 5, price_item)
            
            # 盈亏
            profit = position.get('profit', 0.0)
            profit_item = QTableWidgetItem(f"{profit:.2f}")
            profit_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            # 设置盈亏颜色
            if profit > 0:
                profit_item.setForeground(QBrush(QColor(255, 0, 0)))
            elif profit < 0:
                profit_item.setForeground(QBrush(QColor(0, 255, 0)))
                
            self.position_table.setItem(row, 6, profit_item)


class OrderHistoryWidget(QWidget):
    """委托历史组件"""
    
    def __init__(self, trading_controller, parent=None):
        super().__init__(parent)
        self.trading_controller = trading_controller
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建控制区域
        control_layout = QHBoxLayout()
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新")
        control_layout.addWidget(self.refresh_button)
        
        # 撤单按钮
        self.cancel_order_button = QPushButton("撤单")
        control_layout.addWidget(self.cancel_order_button)
        
        # 添加控制区域到主布局
        main_layout.addLayout(control_layout)
        
        # 创建委托表格
        self.order_table = QTableWidget()
        self.order_table.setColumnCount(7)
        self.order_table.setHorizontalHeaderLabels(["委托时间", "代码", "名称", "方向", "价格", "数量", "状态"])
        self.order_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.order_table.verticalHeader().setVisible(False)
        self.order_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.order_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # 添加表格到主布局
        main_layout.addWidget(self.order_table)
        
        # 连接信号和槽
        self.refresh_button.clicked.connect(self._refresh_orders)
    
    def _refresh_orders(self):
        """刷新委托数据"""
        # TODO: 获取并更新委托数据
        pass
    
    def update_orders(self, orders):
        """更新委托数据"""
        # 清空表格
        self.order_table.setRowCount(0)
        
        # 检查是否有数据
        if not orders or not isinstance(orders, list):
            return
        
        # 设置行数
        self.order_table.setRowCount(len(orders))
        
        # 填充数据
        for row, order in enumerate(orders):
            # 委托时间
            time_item = QTableWidgetItem(order.get('time', ''))
            self.order_table.setItem(row, 0, time_item)
            
            # 代码
            code_item = QTableWidgetItem(order.get('code', ''))
            self.order_table.setItem(row, 1, code_item)
            
            # 名称
            name_item = QTableWidgetItem(order.get('name', ''))
            self.order_table.setItem(row, 2, name_item)
            
            # 方向
            direction = order.get('direction', '')
            direction_item = QTableWidgetItem(direction)
            
            # 设置方向颜色
            if direction == '买入':
                direction_item.setForeground(QBrush(QColor(255, 0, 0)))
            elif direction == '卖出':
                direction_item.setForeground(QBrush(QColor(0, 255, 0)))
                
            self.order_table.setItem(row, 3, direction_item)
            
            # 价格
            price_item = QTableWidgetItem(f"{order.get('price', 0.0):.2f}")
            price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.order_table.setItem(row, 4, price_item)
            
            # 数量
            quantity_item = QTableWidgetItem(str(order.get('quantity', 0)))
            quantity_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.order_table.setItem(row, 5, quantity_item)
            
            # 状态
            status_item = QTableWidgetItem(order.get('status', ''))
            self.order_table.setItem(row, 6, status_item)


class TradingView(QWidget):
    """交易视图"""
    
    def __init__(self, trading_controller, parent=None):
        super().__init__(parent)
        self.trading_controller = trading_controller
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
        
        # 创建下单选项卡
        self.order_widget = OrderWidget(self.trading_controller)
        self.tab_widget.addTab(self.order_widget, "下单")
        
        # 创建持仓选项卡
        self.position_widget = PositionTableWidget(self.trading_controller)
        self.tab_widget.addTab(self.position_widget, "持仓")
        
        # 创建委托选项卡
        self.order_history_widget = OrderHistoryWidget(self.trading_controller)
        self.tab_widget.addTab(self.order_history_widget, "委托")
    
    def initialize_with_data(self, data):
        """使用数据初始化视图"""
        # 更新持仓数据
        positions = data.get("positions", [])
        self.position_widget.update_positions(positions)
        
        # 更新委托数据
        orders = data.get("orders", [])
        self.order_history_widget.update_orders(orders) 
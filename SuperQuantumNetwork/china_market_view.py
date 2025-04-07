#!/usr/bin/env python3
"""
超神系统 - 中国市场分析视图
高级版本，适用于超神桌面系统
"""

import logging
import traceback
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QTableWidget, QTableWidgetItem, 
                           QHeaderView, QGroupBox, QMessageBox, QProgressBar,
                           QSplitter, QFrame, QComboBox, QSpacerItem, QSizePolicy,
                           QCheckBox)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon
import os
import sys
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)


class ChinaMarketWidget(QWidget):
    """超神系统中国市场分析组件 - 高级版"""
    
    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.init_ui()
        
        # 设置对象名，方便样式表定位
        self.setObjectName("chinaMarketWidget")
        
        # 初始刷新
        QTimer.singleShot(500, self.refresh_data)
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 创建顶部面板 - 显示市场指数
        self.create_market_index_panel()
        
        # 创建左右分栏
        self.splitter = QSplitter(Qt.Horizontal)
        
        # 创建左侧面板 - 热点板块和北向资金
        self.create_left_panel()
        
        # 创建右侧面板 - 个股推荐列表
        self.create_right_panel()
        
        # 添加分栏到主布局
        self.main_layout.addWidget(self.splitter)
        
        # 创建底部控制面板
        self.create_bottom_panel()
    
    def create_market_index_panel(self):
        """创建市场指数面板"""
        index_group = QGroupBox("市场指数")
        index_layout = QVBoxLayout(index_group)
        
        # 创建指数信息面板
        index_info_layout = QHBoxLayout()
        
        # 上证指数
        sh_box = QGroupBox("上证指数")
        sh_box.setProperty("class", "index-box")
        sh_layout = QVBoxLayout(sh_box)
        self.sh_value_label = QLabel("--")
        self.sh_value_label.setAlignment(Qt.AlignCenter)
        self.sh_value_label.setProperty("class", "index-value")
        self.sh_change_label = QLabel("--")
        self.sh_change_label.setAlignment(Qt.AlignCenter)
        sh_layout.addWidget(self.sh_value_label)
        sh_layout.addWidget(self.sh_change_label)
        
        # 深证成指
        sz_box = QGroupBox("深证成指")
        sz_box.setProperty("class", "index-box")
        sz_layout = QVBoxLayout(sz_box)
        self.sz_value_label = QLabel("--")
        self.sz_value_label.setAlignment(Qt.AlignCenter)
        self.sz_value_label.setProperty("class", "index-value")
        self.sz_change_label = QLabel("--")
        self.sz_change_label.setAlignment(Qt.AlignCenter)
        sz_layout.addWidget(self.sz_value_label)
        sz_layout.addWidget(self.sz_change_label)
        
        # 创业板指
        cyb_box = QGroupBox("创业板指")
        cyb_box.setProperty("class", "index-box")
        cyb_layout = QVBoxLayout(cyb_box)
        self.cyb_value_label = QLabel("--")
        self.cyb_value_label.setAlignment(Qt.AlignCenter)
        self.cyb_value_label.setProperty("class", "index-value")
        self.cyb_change_label = QLabel("--")
        self.cyb_change_label.setAlignment(Qt.AlignCenter)
        cyb_layout.addWidget(self.cyb_value_label)
        cyb_layout.addWidget(self.cyb_change_label)
        
        # 添加到布局
        index_info_layout.addWidget(sh_box)
        index_info_layout.addWidget(sz_box)
        index_info_layout.addWidget(cyb_box)
        
        # 添加到指数面板
        index_layout.addLayout(index_info_layout)
        
        # 市场风险和趋势
        risk_layout = QHBoxLayout()
        
        # 风险评级
        self.risk_label = QLabel("市场风险: --")
        self.risk_label.setProperty("class", "risk-label")
        risk_layout.addWidget(self.risk_label)
        
        # 市场趋势
        self.trend_label = QLabel("趋势: --")
        self.trend_label.setProperty("class", "trend-label")
        risk_layout.addWidget(self.trend_label)
        
        # 最后更新时间
        self.update_time_label = QLabel("最后更新: --")
        risk_layout.addWidget(self.update_time_label)
        
        index_layout.addLayout(risk_layout)
        
        # 添加到主布局
        self.main_layout.addWidget(index_group)
    
    def create_left_panel(self):
        """创建左侧面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 热点板块
        hot_group = QGroupBox("热点板块")
        hot_layout = QVBoxLayout(hot_group)
        self.sector_table = QTableWidget(0, 2)
        self.sector_table.setHorizontalHeaderLabels(["板块", "强度"])
        self.sector_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.sector_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.sector_table.setEditTriggers(QTableWidget.NoEditTriggers)
        hot_layout.addWidget(self.sector_table)
        
        # 下一轮热点预测
        next_group = QGroupBox("下一轮潜在热点")
        next_layout = QVBoxLayout(next_group)
        self.next_sector_table = QTableWidget(0, 1)
        self.next_sector_table.setHorizontalHeaderLabels(["板块"])
        self.next_sector_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.next_sector_table.setEditTriggers(QTableWidget.NoEditTriggers)
        next_layout.addWidget(self.next_sector_table)
        
        # 北向资金
        north_group = QGroupBox("北向资金")
        north_layout = QVBoxLayout(north_group)
        self.north_flow_label = QLabel("今日净流入: --")
        self.north_flow_5d_label = QLabel("5日净流入: --")
        self.north_trend_label = QLabel("资金趋势: --")
        north_layout.addWidget(self.north_flow_label)
        north_layout.addWidget(self.north_flow_5d_label)
        north_layout.addWidget(self.north_trend_label)
        
        # 添加到左侧布局
        left_layout.addWidget(hot_group)
        left_layout.addWidget(next_group)
        left_layout.addWidget(north_group)
        
        # 添加到分栏
        self.splitter.addWidget(left_widget)
    
    def create_right_panel(self):
        """创建右侧面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 个股推荐
        stock_group = QGroupBox("量子预测个股推荐")
        stock_layout = QVBoxLayout(stock_group)
        
        # 创建表格
        self.stock_table = QTableWidget(0, 6)
        self.stock_table.setHorizontalHeaderLabels(["代码", "名称", "行业", "操作", "当前价", "风险"])
        self.stock_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stock_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stock_table.setAlternatingRowColors(True)
        stock_layout.addWidget(self.stock_table)
        
        # 投资建议
        advice_layout = QHBoxLayout()
        self.position_label = QLabel("建议仓位: --")
        self.position_label.setProperty("class", "advice-label")
        advice_layout.addWidget(self.position_label)
        
        stock_layout.addLayout(advice_layout)
        
        # 添加到右侧布局
        right_layout.addWidget(stock_group)
        
        # 添加到分栏
        self.splitter.addWidget(right_widget)
    
    def create_bottom_panel(self):
        """创建底部控制面板"""
        bottom_layout = QHBoxLayout()
        
        # 操作按钮
        self.refresh_button = QPushButton("刷新数据")
        self.refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.refresh_button.clicked.connect(self.refresh_data)
        
        self.predict_button = QPushButton("超神预测")
        self.predict_button.setIcon(QIcon.fromTheme("system-run"))
        self.predict_button.clicked.connect(self.run_prediction)
        
        # 其他控制
        self.auto_refresh = QCheckBox("自动刷新")
        self.auto_refresh.setChecked(False)
        self.auto_refresh.stateChanged.connect(self.toggle_auto_refresh)
        
        self.refresh_interval = QComboBox()
        self.refresh_interval.addItems(["1分钟", "5分钟", "10分钟", "30分钟"])
        self.refresh_interval.setCurrentIndex(1)  # 默认5分钟
        
        # 添加到布局
        bottom_layout.addWidget(self.refresh_button)
        bottom_layout.addWidget(self.predict_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.auto_refresh)
        bottom_layout.addWidget(self.refresh_interval)
        
        # 添加到主布局
        self.main_layout.addLayout(bottom_layout)
        
        # 自动刷新定时器
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
    
    def toggle_auto_refresh(self, state):
        """切换自动刷新状态"""
        if state == Qt.Checked:
            # 获取刷新间隔（分钟）
            interval_text = self.refresh_interval.currentText()
            interval_min = int(interval_text.split("分")[0])
            interval_ms = interval_min * 60 * 1000
            
            # 启动定时器
            self.refresh_timer.start(interval_ms)
        else:
            # 停止定时器
            self.refresh_timer.stop()
    
    def refresh_data(self):
        """刷新市场数据"""
        if not self.controller:
            return
            
        try:
            # 显示刷新状态
            self.refresh_button.setEnabled(False)
            self.refresh_button.setText("刷新中...")
            
            # 更新市场数据
            self.controller.update_market_data()
            
            # 获取市场数据
            market_data = self.controller.market_data
            
            # 更新指数显示
            self._update_index_display(market_data)
            
            # 更新北向资金
            self._update_north_flow()
            
            # 恢复按钮状态
            self.refresh_button.setEnabled(True)
            self.refresh_button.setText("刷新数据")
            
            # 更新时间
            from datetime import datetime
            self.update_time_label.setText(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"刷新市场数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 恢复按钮状态
            self.refresh_button.setEnabled(True)
            self.refresh_button.setText("刷新数据")
            
            # 显示错误消息
            QMessageBox.warning(self, "刷新失败", f"刷新市场数据失败: {str(e)}")
    
    def _update_index_display(self, market_data):
        """更新指数显示"""
        # 上证指数
        sh_data = market_data.get('sh_index', {})
        sh_close = sh_data.get('close', 0)
        sh_change = sh_data.get('change_pct', 0)
        
        if sh_change is None or (isinstance(sh_change, float) and np.isnan(sh_change)):
            sh_change = 0
            
        self.sh_value_label.setText(f"{sh_close:.2f}")
        self.sh_change_label.setText(f"{sh_change:+.2f}%")
        
        if sh_change >= 0:
            self.sh_change_label.setStyleSheet("color: #FF4444;")
        else:
            self.sh_change_label.setStyleSheet("color: #44FF44;")
        
        # 深证成指
        sz_data = market_data.get('sz_index', {})
        sz_close = sz_data.get('close', 0)
        sz_change = sz_data.get('change_pct', 0)
        
        if sz_change is None or (isinstance(sz_change, float) and np.isnan(sz_change)):
            sz_change = 0
            
        self.sz_value_label.setText(f"{sz_close:.2f}")
        self.sz_change_label.setText(f"{sz_change:+.2f}%")
        
        if sz_change >= 0:
            self.sz_change_label.setStyleSheet("color: #FF4444;")
        else:
            self.sz_change_label.setStyleSheet("color: #44FF44;")
        
        # 创业板指
        cyb_data = market_data.get('cyb_index', {})
        cyb_close = cyb_data.get('close', 0)
        cyb_change = cyb_data.get('change_pct', 0)
        
        if cyb_change is None or (isinstance(cyb_change, float) and np.isnan(cyb_change)):
            cyb_change = 0
            
        self.cyb_value_label.setText(f"{cyb_close:.2f}")
        self.cyb_change_label.setText(f"{cyb_change:+.2f}%")
        
        if cyb_change >= 0:
            self.cyb_change_label.setStyleSheet("color: #FF4444;")
        else:
            self.cyb_change_label.setStyleSheet("color: #44FF44;")
    
    def _update_north_flow(self):
        """更新北向资金数据"""
        try:
            # 获取北向资金数据
            prediction = getattr(self.controller, 'latest_prediction', {})
            north_flow = prediction.get('north_flow', {})
            
            if north_flow:
                # 今日净流入
                daily_flow = north_flow.get('total_inflow', 0) / 100000000  # 单位：亿
                self.north_flow_label.setText(f"今日净流入: {daily_flow:.2f}亿")
                
                # 5日净流入
                flow_5d = north_flow.get('total_flow_5d', 0) / 100000000  # 单位：亿
                self.north_flow_5d_label.setText(f"5日净流入: {flow_5d:.2f}亿")
                
                # 趋势
                trend = north_flow.get('flow_trend', 'unknown')
                self.north_trend_label.setText(f"资金趋势: {trend}")
                
                # 根据正负设置颜色
                if daily_flow > 0:
                    self.north_flow_label.setStyleSheet("color: #FF4444;")
                else:
                    self.north_flow_label.setStyleSheet("color: #44FF44;")
                    
                if flow_5d > 0:
                    self.north_flow_5d_label.setStyleSheet("color: #FF4444;")
                else:
                    self.north_flow_5d_label.setStyleSheet("color: #44FF44;")
        except Exception as e:
            logger.error(f"更新北向资金失败: {str(e)}")
    
    def run_prediction(self):
        """运行市场预测"""
        if not self.controller:
            return
            
        try:
            # 修改按钮状态
            self.predict_button.setEnabled(False)
            self.predict_button.setText("预测中...")
            
            # 运行预测
            prediction = self.controller.predict_market_trend()
            
            if prediction:
                # 获取预测结果
                sector_rotation = prediction.get('sector_rotation', {})
                risk_analysis = prediction.get('risk_analysis', {})
                
                # 获取投资组合建议
                portfolio = self.controller.get_portfolio_suggestion()
                
                # 更新热点板块
                self._update_hot_sectors(sector_rotation)
                
                # 更新风险评级
                overall_risk = risk_analysis.get('overall_risk', 0)
                risk_trend = risk_analysis.get('risk_trend', '')
                self.risk_label.setText(f"市场风险: {overall_risk:.2f}")
                self.trend_label.setText(f"趋势: {risk_trend}")
                
                # 更新建议仓位
                position = portfolio.get('max_position', 0) * 100
                self.position_label.setText(f"建议仓位: {position:.0f}%")
                
                # 更新个股推荐
                self._update_stock_recommendations(portfolio)
                
                # 显示预测完成消息
                QMessageBox.information(self, "预测完成", 
                                      f"超神量子预测完成！\n市场风险评级: {overall_risk:.2f}\n趋势: {risk_trend}")
            
            # 恢复按钮状态
            self.predict_button.setEnabled(True)
            self.predict_button.setText("超神预测")
            
        except Exception as e:
            logger.error(f"运行预测失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 恢复按钮状态
            self.predict_button.setEnabled(True)
            self.predict_button.setText("超神预测")
            
            # 显示错误消息
            QMessageBox.warning(self, "预测失败", f"运行市场预测失败: {str(e)}")
    
    def _update_hot_sectors(self, sector_rotation):
        """更新热点板块"""
        # 当前热点
        hot_sectors = sector_rotation.get('current_hot_sectors', [])
        
        # 清空表格
        self.sector_table.setRowCount(0)
        
        # 添加热点板块
        for i, sector in enumerate(hot_sectors):
            self.sector_table.insertRow(i)
            self.sector_table.setItem(i, 0, QTableWidgetItem(sector))
            
            # 强度 - 使用简单的星级
            strength = QTableWidgetItem("⭐" * (5 - i))  # 越靠前的热度越高
            self.sector_table.setItem(i, 1, strength)
        
        # 下一轮热点
        next_sectors = sector_rotation.get('next_sectors_prediction', [])
        
        # 清空表格
        self.next_sector_table.setRowCount(0)
        
        # 添加下一轮热点
        for i, sector in enumerate(next_sectors):
            self.next_sector_table.insertRow(i)
            self.next_sector_table.setItem(i, 0, QTableWidgetItem(sector))
    
    def _update_stock_recommendations(self, portfolio):
        """更新个股推荐"""
        # 获取推荐列表
        stock_suggestions = portfolio.get('stock_suggestions', [])
        
        # 清空表格
        self.stock_table.setRowCount(0)
        
        # 添加推荐股票
        for i, stock in enumerate(stock_suggestions):
            self.stock_table.insertRow(i)
            
            # 股票代码
            code_item = QTableWidgetItem(stock.get('stock', ''))
            self.stock_table.setItem(i, 0, code_item)
            
            # 股票名称
            name_item = QTableWidgetItem(stock.get('name', ''))
            self.stock_table.setItem(i, 1, name_item)
            
            # 行业
            sector_item = QTableWidgetItem(stock.get('sector', ''))
            self.stock_table.setItem(i, 2, sector_item)
            
            # 操作建议
            action = stock.get('action', '')
            action_item = QTableWidgetItem(action)
            
            # 根据操作设置颜色
            if '买入' in action:
                action_item.setBackground(QColor('#334433'))
                action_item.setForeground(QColor('#55FF55'))
            elif '卖出' in action:
                action_item.setBackground(QColor('#443333'))
                action_item.setForeground(QColor('#FF5555'))
                
            self.stock_table.setItem(i, 3, action_item)
            
            # 当前价格
            price_item = QTableWidgetItem(f"{stock.get('current_price', 0):.2f}")
            self.stock_table.setItem(i, 4, price_item)
            
            # 风险评级
            risk_item = QTableWidgetItem(stock.get('risk_level', ''))
            
            # 根据风险设置颜色
            risk_level = stock.get('risk_level', '')
            if '低' in risk_level:
                risk_item.setForeground(QColor('#44FF44'))
            elif '高' in risk_level:
                risk_item.setForeground(QColor('#FF4444'))
            else:
                risk_item.setForeground(QColor('#FFFF44'))
                
            self.stock_table.setItem(i, 5, risk_item)


def create_market_view(main_window, tab_widget, controller=None):
    """工厂函数：创建中国市场视图"""
    try:
        # 创建标签页
        market_tab = QWidget()
        tab_widget.addTab(market_tab, "中国市场分析")
        
        # 设置布局
        layout = QVBoxLayout(market_tab)
        
        # 创建市场组件
        market_widget = ChinaMarketWidget(parent=market_tab, controller=controller)
        layout.addWidget(market_widget)
        
        return True
    except Exception as e:
        logger.error(f"创建中国市场视图失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # 独立测试
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("中国市场分析视图测试")
    layout = QVBoxLayout(window)
    
    widget = ChinaMarketWidget(window)
    layout.addWidget(widget)
    
    window.setMinimumSize(1024, 768)
    window.show()
    
    sys.exit(app.exec_()) 
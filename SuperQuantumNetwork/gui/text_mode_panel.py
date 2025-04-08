#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神系统 - 纯文本模式面板
基于run_text_mode.py封装的PyQt组件，提供纯文本显示的市场分析和量子决策信息
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QScrollArea, 
    QLabel, QFrame, QSplitter, QHBoxLayout, QPushButton,
    QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QDateTime, QTime
from PyQt5.QtGui import QFont, QColor, QTextCursor, QTextCharFormat, QPalette

import os
import sys
import json
import time
import random
from datetime import datetime
import logging
import math

# ASCII 艺术标题
TITLE = """
    ███████╗██╗   ██╗██████╗ ███████╗██████╗  ██████╗  ██████╗ ██████╗ 
    ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██╔════╝ ██╔═══██╗██╔══██╗
    ███████╗██║   ██║██████╔╝█████╗  ██████╔╝██║  ███╗██║   ██║██║  ██║
    ╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██║   ██║██║   ██║██║  ██║
    ███████║╚██████╔╝██║     ███████╗██║  ██║╚██████╔╝╚██████╔╝██████╔╝
    ╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═════╝ 

              超神系统 v3.5 豪华版 - 纯文本模式界面
"""

# 模拟市场数据
MARKET_INDICES = [
    {"name": "上证指数", "code": "000001", "current": 3288.45, "change": 0.76, "volume": 321.5},
    {"name": "深证成指", "code": "399001", "current": 10573.89, "change": 1.25, "volume": 456.2},
    {"name": "创业板指", "code": "399006", "current": 2105.32, "change": 1.89, "volume": 187.3},
    {"name": "沪深300", "code": "000300", "current": 4027.56, "change": 0.92, "volume": 278.4},
    {"name": "中证500", "code": "000905", "current": 6688.73, "change": 1.13, "volume": 195.7}
]

# 模拟行业板块数据
SECTOR_DATA = [
    {"name": "半导体", "change": 2.87, "recommendation": "强烈推荐", "confidence": 0.92},
    {"name": "新能源", "change": 1.93, "recommendation": "推荐", "confidence": 0.85},
    {"name": "医药生物", "change": 0.76, "recommendation": "中性", "confidence": 0.62},
    {"name": "金融", "change": -0.23, "recommendation": "谨慎", "confidence": 0.58},
    {"name": "消费电子", "change": 2.15, "recommendation": "推荐", "confidence": 0.79},
    {"name": "人工智能", "change": 3.42, "recommendation": "强烈推荐", "confidence": 0.94},
    {"name": "云计算", "change": 1.87, "recommendation": "推荐", "confidence": 0.81}
]

# 模拟北向资金数据
NORTHBOUND_FLOW = {
    "today": 25.76,  # 单位：亿元
    "week": 78.35,
    "month": 234.67,
    "sectors": [
        {"name": "半导体", "amount": 8.45},
        {"name": "新能源", "amount": 6.32},
        {"name": "人工智能", "amount": 5.87}
    ]
}

# 模拟量子网络预测数据
QUANTUM_PREDICTIONS = {
    "market_trend": {
        "short_term": "上涨",
        "mid_term": "震荡上行",
        "confidence": 0.87
    },
    "sector_rotation": [
        {"from": "金融", "to": "科技", "strength": "强"},
        {"from": "消费", "to": "新能源", "strength": "中"}
    ],
    "signals": [
        {"symbol": "600000", "name": "浦发银行", "action": "卖出", "confidence": 0.82},
        {"symbol": "000001", "name": "平安银行", "action": "持有", "confidence": 0.65},
        {"symbol": "002230", "name": "科大讯飞", "action": "买入", "confidence": 0.91},
        {"symbol": "300750", "name": "宁德时代", "action": "买入", "confidence": 0.88},
        {"symbol": "600519", "name": "贵州茅台", "action": "持有", "confidence": 0.72}
    ]
}

class TextModePanel(QWidget):
    """纯文本模式面板，将纯文本输出集成到Qt界面中"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置字体 - 默认大小，将在resize事件中调整
        self.mono_font = QFont("Courier New", 12)
        self.mono_font.setStyleHint(QFont.Monospace)
        self.mono_font.setBold(True)  # 设置为粗体，增强可读性
        
        # 记录当前字体大小
        self.current_font_size = 12
        self.min_font_size = 8
        self.max_font_size = 24
        
        # 创建布局
        self.layout = QVBoxLayout(self)
        
        # 创建文本显示区域
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(self.mono_font)
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: #000020;  /* 更深的蓝黑色背景 */
                color: #FFFFFF;
                border: 2px solid #5050A0;  /* 更粗的边框 */
                padding: 15px;  /* 更大的内边距 */
            }
        """)
        
        # 创建控制区域
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_frame.setStyleSheet("background-color: #1A1A30; border: 1px solid #3A3A5A;")
        control_layout = QHBoxLayout(control_frame)
        
        # 添加刷新按钮
        self.refresh_button = QPushButton("刷新数据")
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #303060;
                color: white;
                border: 1px solid #5050A0;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4040A0;
            }
            QPushButton:pressed {
                background-color: #202040;
            }
        """)
        self.refresh_button.clicked.connect(self.update_display)
        control_layout.addWidget(self.refresh_button)
        
        # 添加字体调整按钮
        self.font_smaller_btn = QPushButton("A-")
        self.font_smaller_btn.setStyleSheet("""
            QPushButton {
                background-color: #303060;
                color: white;
                border: 1px solid #5050A0;
                padding: 5px 10px;
                font-weight: bold;
            }
        """)
        self.font_smaller_btn.clicked.connect(self.decrease_font_size)
        control_layout.addWidget(self.font_smaller_btn)
        
        self.font_larger_btn = QPushButton("A+")
        self.font_larger_btn.setStyleSheet("""
            QPushButton {
                background-color: #303060;
                color: white;
                border: 1px solid #5050A0;
                padding: 5px 10px;
                font-weight: bold;
            }
        """)
        self.font_larger_btn.clicked.connect(self.increase_font_size)
        control_layout.addWidget(self.font_larger_btn)
        
        # 添加自动滚动控制
        self.auto_scroll_btn = QPushButton("自动滚动")
        self.auto_scroll_btn.setCheckable(True)
        self.auto_scroll_btn.setChecked(False)
        self.auto_scroll_btn.setStyleSheet("""
            QPushButton {
                background-color: #303060;
                color: white;
                border: 1px solid #5050A0;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #505080;
            }
        """)
        self.auto_scroll_btn.clicked.connect(self.toggle_auto_scroll)
        control_layout.addWidget(self.auto_scroll_btn)
        
        # 添加状态标签
        self.status_label = QLabel("上次更新: 未更新")
        self.status_label.setStyleSheet("color: #AAAAFF;")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        
        # 添加到主布局
        self.layout.addWidget(self.text_display, 1)
        self.layout.addWidget(control_frame)
        
        # 设置更新计时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(60000)  # 每分钟更新一次
        
        # 自动调整字体的定时器
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.adjust_font_size)
        
        # 自动滚动定时器
        self.auto_scroll_timer = QTimer(self)
        self.auto_scroll_timer.timeout.connect(self.scroll_text)
        self.auto_scroll_active = False
        self.scroll_position = 0
        self.scroll_speed = 2  # 每次滚动的像素数
        
        # 初始化显示
        self.update_display()
    
    def toggle_auto_scroll(self):
        """切换自动滚动状态"""
        self.auto_scroll_active = self.auto_scroll_btn.isChecked()
        if self.auto_scroll_active:
            self.auto_scroll_timer.start(50)  # 50毫秒滚动一次
            self.scroll_position = 0
            self.text_display.verticalScrollBar().setValue(0)
        else:
            self.auto_scroll_timer.stop()
    
    def scroll_text(self):
        """执行文本滚动"""
        if not self.auto_scroll_active:
            return
            
        scrollbar = self.text_display.verticalScrollBar()
        max_value = scrollbar.maximum()
        
        # 如果已经滚动到底部，重新开始
        if self.scroll_position >= max_value:
            self.scroll_position = 0
            scrollbar.setValue(0)
            # 暂停一下再继续滚动
            QTimer.singleShot(3000, self.resume_scroll)
            self.auto_scroll_timer.stop()
        else:
            self.scroll_position += self.scroll_speed
            scrollbar.setValue(self.scroll_position)
    
    def resume_scroll(self):
        """恢复滚动"""
        if self.auto_scroll_active:
            self.auto_scroll_timer.start(50)
    
    def resizeEvent(self, event):
        """窗口大小改变时调整字体大小"""
        super().resizeEvent(event)
        # 启动定时器，避免频繁调整
        self.resize_timer.start(200)
    
    def adjust_font_size(self):
        """根据窗口大小自动调整字体大小"""
        # 获取文本显示区域的大小
        width = self.text_display.width()
        height = self.text_display.height()
        
        # 基于宽度和高度计算合适的字体大小
        # 改进公式，同时考虑宽度和高度
        width_based_size = max(self.min_font_size, min(self.max_font_size, int(width / 80)))
        height_based_size = max(self.min_font_size, min(self.max_font_size, int(height / 50)))
        new_size = min(width_based_size, height_based_size)
        
        # 如果字体大小变化较大，才进行调整
        if abs(new_size - self.current_font_size) >= 1:
            self.current_font_size = new_size
            self.update_font_size(new_size)
    
    def update_font_size(self, size):
        """更新字体大小并刷新显示"""
        # 更新字体
        self.mono_font.setPointSize(size)
        self.text_display.setFont(self.mono_font)
        
        # 更新标题字体大小
        self.title_font_size = size + 2
        
        # 刷新显示
        self.update_display()
    
    def increase_font_size(self):
        """增加字体大小"""
        if self.current_font_size < self.max_font_size:
            self.current_font_size += 1
            self.update_font_size(self.current_font_size)
    
    def decrease_font_size(self):
        """减小字体大小"""
        if self.current_font_size > self.min_font_size:
            self.current_font_size -= 1
            self.update_font_size(self.current_font_size)
    
    def append_colored_text(self, text, color="#FFFFFF"):
        """添加彩色文本到显示区域"""
        cursor = self.text_display.textCursor()
        format = QTextCharFormat()
        
        # 增强颜色 - 让颜色更鲜艳，提高对比度
        bright_colors = {
            "#FFFFFF": "#FFFFFF",     # 白色保持不变
            "#00FF00": "#00FF00",     # 亮绿色保持不变
            "#FF5050": "#FF0000",     # 红色改为更鲜艳的红色
            "#FFFF00": "#FFFF00",     # 黄色保持不变
            "#82AAFF": "#00AAFF",     # 蓝色改为更鲜艳的蓝色
            "#FF00FF": "#FF00FF",     # 紫色保持不变
            "#00FFFF": "#00FFFF",     # 青色保持不变
            "#FFCB6B": "#FFD700"      # 更鲜艳的金色
        }
        
        # 如果有匹配的增强颜色，使用增强的颜色
        enhanced_color = bright_colors.get(color, color)
        
        format.setForeground(QColor(enhanced_color))
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text, format)
        self.text_display.setTextCursor(cursor)
        self.text_display.ensureCursorVisible()
    
    def update_display(self):
        """更新显示内容"""
        self.text_display.clear()
        
        # 添加标题 - 使用数字矩阵风格
        self.append_colored_text("""
 ▀█▀ █░█ █▀▀   █▀▀ █░█ █▀█ █▀▀ █▀█ █▀▀ █▀█ █▀▄
 ░█░ █▀█ █▀▀   ▀▀█ █░█ █▀▀ █▀▀ █▀▄ █▄█ █▄█ █░█
 ▀▀▀ ▀░▀ ▀▀▀   ▀▀▀ ▀▀▀ ▀░░ ▀▀▀ ▀░▀ ▀░▀ ▀░▀ ▀▀░
 ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        QUANTUM SYMBIOTIC SYSTEM V3.5
 ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
""", "#00FFFF")

        now = datetime.now()
        
        # 系统状态 - 采用矩阵式状态显示
        self.append_colored_text("\n", "#FFFFFF")
        self.append_colored_text("// 系统状态矩阵 ", "#00AAFF")
        self.append_colored_text("[CONNECTED]", "#00FF00")
        self.append_colored_text("\n", "#FFFFFF")
        
        # 使用ASCII艺术制作系统状态矩阵
        status_matrix = [
            "██████ ",
            "██████ ",
            "██████ ",
            "██████ ",
        ]
        
        status_labels = [
            f"系统版本: 超神3.5    ",
            f"时间戳: {now.strftime('%H:%M:%S')}    ",
            f"量子状态: COHERENT    ",
            f"安全等级: MAXIMUM    ",
        ]
        
        for i, (matrix, label) in enumerate(zip(status_matrix, status_labels)):
            color = random.choice(["#00AAFF", "#00FFAA", "#FFAA00", "#AA00FF"])
            self.append_colored_text(matrix, color)
            self.append_colored_text(label, "#FFFFFF")
            
            # 添加随机的矩阵数字效果
            matrix_effect = ''.join([str(random.randint(0, 1)) for _ in range(25)])
            self.append_colored_text(matrix_effect + "\n", "#00FF00")
            
        # 创建分隔符
        self.append_colored_text("\n", "#FFFFFF")
        
        # =========== 全新动态市场脉搏设计 ===========
        self.append_colored_text("// MARKET PULSE ", "#FF9900")
        self.append_colored_text("[ 量子数据流 ]\n\n", "#FFFF00")
        
        # 获取各种数据
        market_trends = ["震荡上行", "短期回调", "趋势不变", "稳中有升"]
        trend = random.choice(market_trends)
        signal_strength = random.randint(75, 95)
        industries = ["新能源", "半导体", "人工智能", "医药", "金融"]
        probs = [random.randint(65, 90) for _ in range(len(industries))]
        max_prob_index = probs.index(max(probs))
        
        # 采用类似代码的形式展示数据
        self.append_colored_text("pulse.scan(market) {\n", "#FF6600")
        
        # 核心指数 - 使用数据流样式
        self.append_colored_text("  core_index: ", "#FFAA00")
        self.append_colored_text(f"{signal_strength}", "#FFFFFF")
        
        # 添加随机的二进制噪声效果
        noise = ''.join(['1' if random.random() > 0.7 else '0' for _ in range(30)])
        self.append_colored_text(f" [0x{hex(signal_strength)[2:].upper()}] ", "#00AAFF")
        self.append_colored_text(f"{noise}\n", "#444444")
        
        # 市场趋势 - 使用数据流样式
        self.append_colored_text("  trend: ", "#FFAA00")
        self.append_colored_text(f"{trend}", "#FFFFFF")
        
        # 趋势可视化 - 使用ASCII艺术
        trend_visual = ""
        if "上行" in trend:
            trend_visual = "↗↗↗↗↗"
            trend_color = "#00FF00"
        elif "回调" in trend:
            trend_visual = "↘↘↘↘↘"
            trend_color = "#FF0000"
        else:
            trend_visual = "→→→→→"
            trend_color = "#FFFF00"
            
        self.append_colored_text(" ")
        self.append_colored_text(trend_visual, trend_color)
        self.append_colored_text("\n", "#FFFFFF")
        
        # 量子信号 - 使用脉冲可视化
        self.append_colored_text("  quantum_signal: ", "#FFAA00")
        
        # 创建动态脉冲图
        pulse = []
        for i in range(20):
            value = int(50 + 30 * math.sin(i/3) + random.randint(-5, 5))
            value = min(90, max(10, value))
            pulse.append(value)
            
        # 绘制脉冲图
        pulse_chars = " ▁▂▃▄▅▆▇█"
        pulse_str = ""
        for p in pulse:
            idx = int(p / 10)
            pulse_str += pulse_chars[idx]
            
        self.append_colored_text(pulse_str, "#00FFFF")
        self.append_colored_text("\n", "#FFFFFF")
        
        # 行业分析 - 使用代码样式
        self.append_colored_text("  sector_analysis: {\n", "#FFAA00")
        
        # 行业数据 - 使用条形图可视化
        for i, (industry, prob) in enumerate(zip(industries, probs)):
            self.append_colored_text(f"    {industry}: ", "#00FFAA")
            
            # 概率值
            self.append_colored_text(f"{prob}", "#FFFFFF")
            
            # 可视化条形图
            bar_len = prob // 5  # 20为满值
            if i == max_prob_index:
                # 突出显示最高概率行业
                self.append_colored_text(" [", "#FFFFFF")
                self.append_colored_text("█" * bar_len, "#FFFF00")
                self.append_colored_text("]", "#FFFFFF")
                self.append_colored_text(" *MAX*", "#FF0000")
            else:
                self.append_colored_text(" [", "#FFFFFF")
                self.append_colored_text("█" * bar_len, "#00AAFF")
                self.append_colored_text("]", "#FFFFFF")
                
            self.append_colored_text("\n", "#FFFFFF")
            
        self.append_colored_text("  }\n", "#FFAA00")
        
        # 数据分析结果
        self.append_colored_text("  analysis_result: ", "#FFAA00")
        suggestions = ["增持科技股", "吸纳优质蓝筹", "关注周期股", "配置消费龙头"]
        suggestion = random.choice(suggestions)
        self.append_colored_text(f"\"{suggestion}\"", "#00FF00")
        
        # 添加置信度
        confidence = random.randint(85, 99)
        self.append_colored_text(f" // 置信度: {confidence}%\n", "#888888")
        
        # 添加随机数据点
        self.append_colored_text("  data_points: ", "#FFAA00")
        data_points = [random.randint(100, 999) for _ in range(8)]
        data_str = ', '.join([str(p) for p in data_points])
        self.append_colored_text(data_str, "#FFFFFF")
        self.append_colored_text("\n", "#FFFFFF")
        
        # 量子纠缠状态
        self.append_colored_text("  entanglement_status: ", "#FFAA00")
        self.append_colored_text("OPTIMAL", "#00FF00")
        self.append_colored_text("\n", "#FFFFFF")
        
        # 收尾
        self.append_colored_text("}\n", "#FF6600")
        
        # 添加脉搏模拟效果
        self.append_colored_text("\n// 量子脉搏模拟\n", "#00AAFF")
        
        # 创建三行随机波形图
        for i in range(3):
            wave = ""
            colors = ["#FF0000", "#00FF00", "#00AAFF"]
            
            # 使用三角函数生成波形
            for x in range(60):
                phase = random.uniform(0, 0.5)
                y = math.sin(x/5 + phase) + math.cos(x/10)
                y = (y + 2) / 4  # 归一化到0-1范围
                
                # 将值映射到字符
                chars = " .-=+*#%@"
                idx = int(y * (len(chars) - 1))
                wave += chars[idx]
                
            self.append_colored_text(wave + "\n", colors[i])
        
        # 添加数据流结束标记
        self.append_colored_text("\n// END OF MARKET PULSE DATA STREAM\n", "#888888")
        
        # =========== 市场指数 - 使用表格式布局 ===========
        self.append_colored_text("\n// MARKET INDICES ", "#00AAFF")
        self.append_colored_text("[ 实时数据 ]\n\n", "#FFFFFF")
        
        # 表头
        self.append_colored_text(" 指数名称      代码      点位      变化      成交量  \n", "#888888")
        
        # 分隔线使用点阵
        self.append_colored_text(" · · · · · · · · · · · · · · · · · · · · · · · · · \n", "#444444")
        
        # 指数数据
        for index in MARKET_INDICES:
            # 使数据有些微变化
            current = index["current"] * (1 + random.uniform(-0.002, 0.002))
            change = index["change"] * (1 + random.uniform(-0.1, 0.1))
            volume = index["volume"] * (1 + random.uniform(-0.05, 0.05))
            
            name = index["name"].ljust(10)
            code = index["code"].ljust(8)
            current_str = f"{current:.2f}".ljust(8)
            volume_str = f"{volume:.1f}".ljust(8)
            
            self.append_colored_text(f" {name} ", "#00AAFF")
            self.append_colored_text(f"{code} ", "#FFFFFF")
            self.append_colored_text(f"{current_str} ", "#FFFFFF")
            
            if change > 0:
                self.append_colored_text(f"+{change:.2f}% ".ljust(10), "#00FF00")
            else:
                self.append_colored_text(f"{change:.2f}% ".ljust(10), "#FF0000")
                
            self.append_colored_text(f"{volume_str}\n", "#FFFFFF")
            
        # =========== 量子预测 - 使用未来主义风格 ===========
        self.append_colored_text("\n// QUANTUM PREDICTIONS ", "#FF00FF")
        self.append_colored_text("[ 概率波函数 ]\n\n", "#FFFFFF")
        
        # 获取预测数据
        qp = QUANTUM_PREDICTIONS
        
        # 预测格式类似于代码
        self.append_colored_text("function quantumPredict() {\n", "#FF00FF")
        
        # 短期预测
        self.append_colored_text("  short_term: ", "#FF00FF")
        self.append_colored_text(f"\"{qp['market_trend']['short_term']}\"", "#FFFFFF")
        
        # 预测时间
        time_frame = random.randint(1, 3)
        self.append_colored_text(f" // T+{time_frame}\n", "#888888")
        
        # 中期预测
        self.append_colored_text("  mid_term: ", "#FF00FF")
        self.append_colored_text(f"\"{qp['market_trend']['mid_term']}\"", "#FFFFFF")
        self.append_colored_text(" // T+7\n", "#888888")
        
        # 概率分布图 - 使用ASCII艺术
        self.append_colored_text("  probability_distribution: [\n", "#FF00FF")
        
        # 生成随机的概率分布
        for i in range(3):
            self.append_colored_text("    ", "#FFFFFF")
            
            # 随机生成概率分布
            for j in range(20):
                p = random.random()
                if p > 0.7:
                    self.append_colored_text("█", "#FF00FF")
                elif p > 0.4:
                    self.append_colored_text("▓", "#AA00FF")
                elif p > 0.2:
                    self.append_colored_text("▒", "#8800FF")
                else:
                    self.append_colored_text("░", "#6600FF")
            
            self.append_colored_text("\n", "#FFFFFF")
            
        self.append_colored_text("  ]\n", "#FF00FF")
        
        # 关键投资点
        self.append_colored_text("  key_signals: [\n", "#FF00FF")
        
        # 显示几个关键信号
        signals = qp["signals"]
        for signal in signals[:3]:  # 取前三个信号
            symbol = signal["symbol"]
            name = signal["name"]
            action = signal["action"]
            confidence = signal["confidence"]
            
            self.append_colored_text(f"    {symbol} ", "#FFFFFF")
            self.append_colored_text(f"// {name} ", "#888888")
            
            if action == "买入":
                self.append_colored_text(f"BUY", "#00FF00")
            elif action == "卖出":
                self.append_colored_text(f"SELL", "#FF0000")
            else:
                self.append_colored_text(f"HOLD", "#FFFF00")
                
            self.append_colored_text(" ")
            
            # 可视化置信度
            conf_bar = int(confidence * 10)
            self.append_colored_text("[", "#FFFFFF")
            self.append_colored_text("■" * conf_bar, "#FF00FF")
            self.append_colored_text("□" * (10 - conf_bar), "#444444")
            self.append_colored_text("]\n", "#FFFFFF")
            
        self.append_colored_text("  ]\n", "#FF00FF")
        
        # 收尾
        self.append_colored_text("}\n", "#FF00FF")
        
        # 添加随机矩阵效果(黑客帝国风格)
        self.append_colored_text("\n// 量子矩阵 //\n", "#00FF00")
        
        for i in range(5):
            matrix_line = ""
            for j in range(60):
                if random.random() > 0.7:
                    matrix_line += random.choice(["0", "1"])
                else:
                    matrix_line += " "
            self.append_colored_text(matrix_line + "\n", "#00FF00")
            
        # 图示系统边缘界面
        self.append_colored_text("\n// SYSTEM BOUNDARY\n", "#FF0000")
        
        # 更新状态标签
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        self.status_label.setText(f"上次更新: {now_str}")
        
        # 滚动到顶部
        self.text_display.moveCursor(QTextCursor.Start)
    
    def append_title(self, title_text, color):
        """添加带格式的标题"""
        # 添加空行
        self.append_colored_text("\n", "#FFFFFF")
        
        # 创建分隔线
        separator = "=" * (len(title_text) + 20)
        
        # 添加标题
        self.append_colored_text(f"{separator}\n", color)
        
        # 设置标题文本格式
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # 创建字体格式
        title_format = QTextCharFormat()
        title_format.setForeground(QColor(color))
        
        # 设置更大的字体 - 使用当前字体大小的放大版本
        title_font = QFont("Courier New", self.current_font_size + 2)
        title_font.setBold(True)
        title_format.setFont(title_font)
        
        # 插入文本
        center_text = " " * 5 + title_text + " " * 5
        cursor.insertText(center_text, title_format)
        self.text_display.setTextCursor(cursor)
        
        # 添加分隔线
        self.append_colored_text(f"\n{separator}\n", color)
        
        # 添加空行
        self.append_colored_text("\n", "#FFFFFF")
    
    def showEvent(self, event):
        """当面板显示时启动计时器"""
        super().showEvent(event)
        self.timer.start()
    
    def hideEvent(self, event):
        """当面板隐藏时停止计时器"""
        super().hideEvent(event)
        self.timer.stop()


if __name__ == "__main__":
    # 独立测试代码
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    panel = TextModePanel()
    panel.resize(1000, 800)
    panel.setWindowTitle("超神系统 - 纯文本模式")
    panel.show()
    
    sys.exit(app.exec_()) 
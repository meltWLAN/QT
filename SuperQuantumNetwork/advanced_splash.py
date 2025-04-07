#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 高级启动画面
提供更现代化的启动体验
"""

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, 
    QGraphicsDropShadowEffect, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QPropertyAnimation, QEasingCurve, QPointF, QRectF
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QLinearGradient, QPen, QBrush, QRadialGradient
import sys
import random
import math

class QuantumParticle:
    """量子粒子类，用于实现粒子动画效果"""
    
    def __init__(self, x, y, speed, size):
        self.x = x
        self.y = y
        self.speed = speed
        self.size = size
        
        # 运动方向 (随机角度)
        self.angle = random.uniform(0, 2 * 3.14159)
        
        # 随机颜色 - 使用更多的蓝色和青色调
        r = random.randint(0, 100)
        g = random.randint(100, 255)
        b = random.randint(200, 255)
        self.color = (r, g, b)
        
        # 粒子生命周期与脉动效果
        self.life = 1.0  # 生命值 (0-1)
        self.life_change = random.uniform(-0.005, 0.005)  # 生命变化率
        
        # 粒子脉动
        self.pulse_size = self.size
        self.pulse_direction = random.choice([-1, 1])
        self.pulse_speed = random.uniform(0.01, 0.05)
        
        # 轨迹历史 (用于绘制尾迹)
        self.history = []
        self.max_history = 8
    
    def update(self):
        """更新粒子位置和状态"""
        # 更新位置
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        # 边界检查 - 回弹或环绕
        if random.random() < 0.7:  # 70%概率回弹
            if self.x < 50 or self.x > 650:
                self.angle = 3.14159 - self.angle  # 水平反弹
            if self.y < 50 or self.y > 450:
                self.angle = -self.angle  # 垂直反弹
        else:  # 30%概率环绕
            if self.x < 0:
                self.x = 700
            elif self.x > 700:
                self.x = 0
            if self.y < 0:
                self.y = 500
            elif self.y > 500:
                self.y = 0
        
        # 稍微随机改变角度 (布朗运动)
        self.angle += random.uniform(-0.1, 0.1)
        
        # 更新生命周期
        self.life += self.life_change
        if self.life > 1.0 or self.life < 0.2:
            self.life_change = -self.life_change
            self.life = max(0.2, min(1.0, self.life))
        
        # 更新脉动
        self.pulse_size += self.pulse_direction * self.pulse_speed
        if self.pulse_size > self.size * 1.3 or self.pulse_size < self.size * 0.7:
            self.pulse_direction = -self.pulse_direction
        
        # 保存位置历史
        self.history.append((self.x, self.y))
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    @property
    def display_size(self):
        """获取当前显示大小 (考虑脉动和生命周期)"""
        return self.pulse_size * self.life

class SuperGodSplashScreen(QWidget):
    """超神启动画面"""
    
    # 自定义信号
    progressChanged = pyqtSignal(int, str)
    finished = pyqtSignal()
    
    def __init__(self):
        super().__init__(None)
        
        # 窗口设置
        self.setWindowTitle("启动中")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(700, 500)
        
        # 初始化粒子
        self.particles = []
        for _ in range(50):  # 增加粒子数量
            x = random.uniform(50, 650)
            y = random.uniform(50, 450)
            speed = random.uniform(0.2, 1.0)  # 提高速度上限
            size = random.uniform(2, 8)  # 增大粒子尺寸
            self.particles.append(QuantumParticle(x, y, speed, size))
        
        # 添加量子能量波动效果
        self.energy_waves = []
        for _ in range(3):
            radius = random.uniform(50, 150)
            speed = random.uniform(0.5, 1.5)
            x = random.uniform(200, 500)
            y = random.uniform(150, 350)
            self.energy_waves.append({"x": x, "y": y, "radius": radius, "max_radius": radius + 100, "speed": speed, "opacity": 0.8})
        
        # 设置UI
        self._setup_ui()
        
        # 动画定时器
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_timer.start(30)  # 30ms刷新一次，约33fps
        
        # 水波纹效果定时器
        self.wave_timer = QTimer(self)
        self.wave_timer.timeout.connect(self._update_waves)
        self.wave_timer.start(50)
        
        # 连接信号
        self.progressChanged.connect(self._update_progress)
        
        # 居中显示
        self._center_on_screen()
    
    def _setup_ui(self):
        """设置UI"""
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 顶部空间
        layout.addSpacing(30)
        
        # 标题
        title_label = QLabel("超神量子共生网络交易系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #00DDFF;")
        
        # 添加高级阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 170, 255, 180))
        shadow.setOffset(0, 0)
        title_label.setGraphicsEffect(shadow)
        
        layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel("SUPER GOD-LEVEL QUANTUM SYMBIOTIC SYSTEM")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setStyleSheet("color: rgba(255, 255, 255, 180);")
        layout.addWidget(subtitle_label)
        
        layout.addSpacing(40)
        
        # 核心技术标签
        tech_frame = QFrame()
        tech_frame.setStyleSheet("background-color: rgba(10, 10, 24, 150); border-radius: 10px;")
        tech_layout = QHBoxLayout(tech_frame)
        
        tech_items = [
            ("量子共生技术", "#00FFCC"),
            ("神经网络学习", "#FFCC00"),
            ("超维度预测", "#FF66AA"),
            ("全息市场分析", "#66CCFF")
        ]
        
        for tech, color in tech_items:
            tech_label = QLabel(tech)
            tech_label.setStyleSheet(f"color: {color}; background-color: rgba(0, 0, 0, 100); padding: 5px 10px; border-radius: 5px;")
            tech_label.setAlignment(Qt.AlignCenter)
            tech_layout.addWidget(tech_label)
        
        layout.addWidget(tech_frame)
        
        layout.addStretch()
        
        # 版本信息
        version_label = QLabel("v2.0.0 超神进化")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: rgba(255, 255, 255, 150);")
        layout.addWidget(version_label)
        
        layout.addSpacing(15)
        
        # 进度文本
        self.status_label = QLabel("正在初始化量子网络...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(20, 20, 40, 150);
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #0055AA, stop:0.5 #00DDFF, stop:1 #0055AA);
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 底部版权信息
        copyright_label = QLabel("© 2023 量子共生网络研发团队")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet("color: rgba(255, 255, 255, 100); font-size: 9px; padding-top: 10px;")
        layout.addWidget(copyright_label)
    
    def _center_on_screen(self):
        """将窗口居中显示"""
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def _update_progress(self, value, message):
        """更新进度条"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
        # 达到100%时发出完成信号
        if value >= 100:
            self.finished.emit()
    
    def _update_animation(self):
        """更新动画效果"""
        # 更新粒子位置
        for particle in self.particles:
            particle.update()
        
        # 重绘界面
        self.update()
    
    def _update_waves(self):
        """更新能量波动效果"""
        for wave in self.energy_waves:
            wave["radius"] += wave["speed"]
            wave["opacity"] -= 0.01
            
            # 超出最大半径时重置
            if wave["radius"] > wave["max_radius"] or wave["opacity"] <= 0:
                wave["radius"] = random.uniform(10, 50)
                wave["opacity"] = 0.8
        
        # 随机添加新波动
        if random.random() < 0.05:  # 5%的概率
            x = random.uniform(200, 500)
            y = random.uniform(150, 350)
            radius = random.uniform(20, 80)
            self.energy_waves.append({
                "x": x, "y": y, 
                "radius": radius, 
                "max_radius": radius + 150, 
                "speed": random.uniform(0.8, 2.0), 
                "opacity": 0.8
            })
            
            # 限制波动数量
            if len(self.energy_waves) > 10:
                self.energy_waves.pop(0)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制半透明背景
        painter.fillRect(self.rect(), QColor(10, 10, 24, 230))
        
        # 绘制能量波动
        for wave in self.energy_waves:
            gradient = QRadialGradient(wave["x"], wave["y"], wave["radius"])
            gradient.setColorAt(0, QColor(0, 170, 255, 0))
            gradient.setColorAt(0.8, QColor(0, 200, 255, int(80 * wave["opacity"])))
            gradient.setColorAt(1, QColor(0, 100, 255, 0))
            
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(wave["x"] - wave["radius"], wave["y"] - wave["radius"], wave["radius"] * 2, wave["radius"] * 2))
        
        # 绘制粒子
        for particle in self.particles:
            # 设置粒子颜色和大小
            color = particle.color
            painter.setBrush(QBrush(QColor(*color)))
            painter.setPen(Qt.NoPen)
            
            # 绘制粒子
            particle_size = particle.display_size
            painter.drawEllipse(QRectF(int(particle.x - particle_size/2), 
                               int(particle.y - particle_size/2), 
                               int(particle_size), int(particle_size)))
            
            # 绘制粒子连接线
            for other_particle in self.nearest_particles(particle, 3):
                # 根据距离计算线的透明度
                distance = self._distance(particle, other_particle)
                max_distance = 200
                alpha = max(10, int(255 * (1 - distance / max_distance)))
                
                if alpha > 0:
                    # 设置线的颜色和宽度
                    pen = QPen(QColor(0, 150, 255, alpha))
                    pen.setWidth(1)
                    painter.setPen(pen)
                    
                    # 绘制连接线
                    painter.drawLine(QPointF(int(particle.x), int(particle.y)),
                                  QPointF(int(other_particle.x), int(other_particle.y)))
        
        # 绘制边缘发光效果
        rect = self.rect()
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor(0, 170, 255, 60))
        gradient.setColorAt(0.5, QColor(0, 100, 255, 20))
        gradient.setColorAt(1, QColor(0, 170, 255, 60))
        
        painter.setPen(QPen(QBrush(gradient), 3))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)
    
    def close(self):
        """关闭窗口时停止计时器"""
        self.animation_timer.stop()
        self.wave_timer.stop()
        super().close()
    
    def nearest_particles(self, particle, count=3):
        """查找最近的粒子"""
        distances = []
        for other in self.particles:
            if other != particle:
                distance = self._distance(particle, other)
                if distance < 200:  # 只考虑距离小于200的粒子
                    distances.append((distance, other))
        
        # 排序并返回最近的n个粒子
        distances.sort(key=lambda x: x[0])
        return [p for _, p in distances[:count]]
    
    def _distance(self, p1, p2):
        """计算两个粒子之间的距离"""
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def show_splash_screen(app):
    """显示启动画面并返回实例"""
    splash = SuperGodSplashScreen()
    splash.show()
    app.processEvents()
    return splash

if __name__ == "__main__":
    # 测试代码
    app = QApplication(sys.argv)
    splash = show_splash_screen(app)
    
    # 模拟加载过程
    def simulate_loading():
        current = 0
        stages = [
            "正在初始化量子网络...",
            "加载市场数据...",
            "校准量子共振频率...",
            "同步交易引擎...",
            "激活AI预测模块..."
        ]
        
        for i, stage in enumerate(stages):
            current = (i+1) * 100 // len(stages)
            splash.progressChanged.emit(current, stage)
            QTimer.singleShot((i+1)*1000, lambda: None)  # 延迟
    
    QTimer.singleShot(100, simulate_loading)
    
    # 主窗口
    main_window = QWidget()
    main_window.setWindowTitle("主窗口")
    main_window.resize(800, 600)
    
    def show_main():
        main_window.show()
    
    splash.finished.connect(show_main)
    
    sys.exit(app.exec_()) 
#!/usr/bin/env python3
"""
超神系统 - 系统验证和诊断工具
"""

import sys
import os
import logging
import time
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger("SuperGodVerifier")

def verify_dependencies():
    """验证依赖库"""
    dependencies = [
        ('PyQt5', '基础GUI库'),
        ('numpy', '数值计算库'),
        ('pandas', '数据分析库'),
        ('pyqtgraph', '高级图形库'),
        ('qtawesome', '图标库'),
        ('qdarkstyle', '暗色主题库'),
        ('qt_material', '材质设计库'),
        ('networkx', '图论库')
    ]
    
    status = True
    for package, description in dependencies:
        try:
            __import__(package)
            logger.info(f"✓ {package} ({description}): 已安装")
        except ImportError as e:
            logger.error(f"✗ {package} ({description}): 未安装 - {str(e)}")
            status = False
    
    return status

def verify_core_modules():
    """验证核心模块"""
    modules = [
        ('market_controllers', 'MarketDataController', '市场数据控制器'),
        ('market_controllers', 'TradeController', '交易控制器'),
        ('quantum_symbiotic_network.core.quantum_entanglement_engine', 'QuantumEntanglementEngine', '量子纠缠引擎'),
        ('dashboard_module', 'create_dashboard', '仪表盘创建函数'),
        ('quantum_view', 'create_quantum_view', '量子视图创建函数'),
        ('advanced_splash', 'SuperGodSplashScreen', '高级启动画面')
    ]
    
    status = True
    for module_name, class_name, description in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            component = getattr(module, class_name)
            logger.info(f"✓ {module_name}.{class_name} ({description}): 已验证")
        except ImportError as e:
            logger.error(f"✗ {module_name}.{class_name} ({description}): 导入失败 - {str(e)}")
            status = False
        except AttributeError as e:
            logger.error(f"✗ {module_name}.{class_name} ({description}): 类/函数不存在 - {str(e)}")
            status = False
    
    return status

def verify_gui_components():
    """验证GUI组件"""
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication([])
        
        # 验证仪表盘
        logger.info("正在验证仪表盘组件...")
        from dashboard_module import create_dashboard
        dashboard = create_dashboard()
        logger.info("✓ 仪表盘组件创建成功")
        
        # 验证量子视图
        logger.info("正在验证量子视图组件...")
        from quantum_view import create_quantum_view
        quantum_view = create_quantum_view()
        logger.info("✓ 量子视图组件创建成功")
        
        # 验证启动画面
        logger.info("正在验证启动画面组件...")
        from advanced_splash import SuperGodSplashScreen
        splash = SuperGodSplashScreen()
        logger.info("✓ 启动画面组件创建成功")
        
        # 验证系统托盘图标修复
        logger.info("正在验证系统托盘图标修复...")
        from PyQt5.QtWidgets import QSystemTrayIcon
        from PyQt5.QtGui import QIcon, QPixmap, QColor
        icon_pixmap = QPixmap(32, 32)
        icon_pixmap.fill(QColor(0, 128, 255))
        tray_icon = QSystemTrayIcon(QIcon(icon_pixmap))
        tray_icon.show()
        logger.info("✓ 系统托盘图标创建成功")
        
        return True
    except Exception as e:
        logger.error(f"GUI组件验证失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def verify_quantum_engine():
    """验证量子引擎"""
    try:
        from quantum_symbiotic_network.core.quantum_entanglement_engine import QuantumEntanglementEngine
        
        # 创建引擎实例
        engine = QuantumEntanglementEngine(dimensions=8, entanglement_factor=0.3)
        logger.info("✓ 量子引擎创建成功")
        
        # 初始化测试实体
        test_entities = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        
        # 创建相关性矩阵
        correlation_matrix = {}
        for i in range(len(test_entities)):
            for j in range(i+1, len(test_entities)):
                entity1 = test_entities[i]
                entity2 = test_entities[j]
                # 生成随机相关性
                correlation = 0.5 + (i * j) % 5 * 0.1
                correlation_matrix[(entity1, entity2)] = correlation
        
        # 初始化纠缠关系
        clusters = engine.initialize_entanglement(test_entities, correlation_matrix)
        logger.info(f"✓ 量子纠缠关系初始化成功，纠缠群组数: {len(clusters)}")
        
        # 测试市场共振计算
        test_market_data = {}
        for entity in test_entities:
            test_market_data[entity] = {
                'price': 100.0 + hash(entity) % 100,
                'price_change_pct': 0.01 * (hash(entity) % 10),
                'volume_relative': 1.0 + 0.1 * (hash(entity) % 5)
            }
        
        resonance = engine.compute_market_resonance(test_market_data)
        logger.info(f"✓ 市场共振计算成功，实体数: {len(resonance)}")
        
        # 测试市场预测
        predictions = engine.predict_market_movement(test_entities)
        logger.info(f"✓ 市场预测成功，预测数: {len(predictions)}")
        
        return True
    except Exception as e:
        logger.error(f"量子引擎验证失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def verify_controllers():
    """验证控制器组件"""
    try:
        from market_controllers import MarketDataController, TradeController
        
        # 创建市场数据控制器
        market_controller = MarketDataController()
        logger.info("✓ 市场数据控制器创建成功")
        
        # 更新市场数据
        result = market_controller.update_market_data()
        logger.info(f"✓ 市场数据更新: {'成功' if result else '失败'}")
        
        # 测试市场预测
        prediction = market_controller.predict_market_trend()
        logger.info(f"✓ 市场趋势预测成功，预测项: {len(prediction)}")
        
        # 创建交易控制器
        trade_controller = TradeController(market_controller=market_controller)
        logger.info("✓ 交易控制器创建成功")
        
        # 测试交易功能
        trade_controller.enable_trading("simulation")
        logger.info(f"✓ 交易功能启用成功，模式: simulation")
        
        # 测试下单功能
        order_id = trade_controller.place_order("000001.SH", "buy", 100)
        logger.info(f"✓ 下单功能测试成功，订单ID: {order_id}")
        
        # 获取持仓
        positions = trade_controller.get_positions()
        logger.info(f"✓ 持仓获取成功，持仓数: {len(positions)}")
        
        return True
    except Exception as e:
        logger.error(f"控制器验证失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("超神系统深度验证工具 v1.0")
    logger.info("=" * 80)
    
    # 验证依赖
    logger.info("\n[1/5] 验证依赖库...")
    dependencies_ok = verify_dependencies()
    
    # 验证核心模块
    logger.info("\n[2/5] 验证核心模块...")
    core_modules_ok = verify_core_modules()
    
    # 验证GUI组件
    logger.info("\n[3/5] 验证GUI组件...")
    gui_ok = verify_gui_components()
    
    # 验证量子引擎
    logger.info("\n[4/5] 验证量子引擎...")
    quantum_ok = verify_quantum_engine()
    
    # 验证控制器
    logger.info("\n[5/5] 验证控制器组件...")
    controllers_ok = verify_controllers()
    
    # 汇总结果
    end_time = time.time()
    elapsed = end_time - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info(f"验证完成，耗时: {elapsed:.2f} 秒")
    logger.info("=" * 80)
    
    logger.info(f"依赖库验证: {'通过' if dependencies_ok else '失败'}")
    logger.info(f"核心模块验证: {'通过' if core_modules_ok else '失败'}")
    logger.info(f"GUI组件验证: {'通过' if gui_ok else '失败'}")
    logger.info(f"量子引擎验证: {'通过' if quantum_ok else '失败'}")
    logger.info(f"控制器验证: {'通过' if controllers_ok else '失败'}")
    
    overall_status = all([dependencies_ok, core_modules_ok, gui_ok, quantum_ok, controllers_ok])
    logger.info(f"\n系统整体状态: {'正常' if overall_status else '异常'}")
    
    return 0 if overall_status else 1

if __name__ == "__main__":
    sys.exit(main()) 
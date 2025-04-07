#!/usr/bin/env python3
"""
超神系统 - 功能测试套件运行脚本
运行所有功能测试并生成测试报告
"""

import os
import sys
import unittest
import logging
import time
import json
from datetime import datetime
import traceback

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, 'logs', f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), 'w')
    ]
)
logger = logging.getLogger("TestRunner")

# 确保日志目录存在
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)


def discover_tests():
    """发现所有测试用例"""
    logger.info("开始发现测试用例...")
    
    # 发现测试目录下的所有测试
    test_loader = unittest.TestLoader()
    
    # 首先加载非GUI测试
    test_suite = test_loader.discover(
        os.path.dirname(script_dir),  # tests目录
        pattern="test_*.py",
        top_level_dir=project_root
    )
    
    # 然后尝试加载GUI测试
    try:
        # 尝试导入PyQt5，如果成功则加载GUI测试
        import PyQt5
        
        # 加载functional目录下的GUI测试
        gui_tests = test_loader.discover(
            script_dir,  # functional目录
            pattern="test_gui.py"
        )
        
        # 合并测试套件
        test_suite.addTests(gui_tests)
        logger.info("GUI测试已加载")
    except ImportError:
        logger.warning("PyQt5未安装，跳过GUI测试")
    
    # 计算测试用例数量
    test_count = test_suite.countTestCases()
    logger.info(f"共发现 {test_count} 个测试用例")
    
    return test_suite


def generate_report(result, start_time, end_time):
    """生成测试报告"""
    runtime = end_time - start_time
    
    # 构建报告数据
    report = {
        "summary": {
            "total": result.testsRun,
            "success": result.testsRun - len(result.errors) - len(result.failures),
            "failure": len(result.failures),
            "error": len(result.errors),
            "runtime": f"{runtime:.2f}秒",
            "success_rate": f"{(result.testsRun - len(result.errors) - len(result.failures)) / result.testsRun * 100:.1f}%",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "failures": [],
        "errors": []
    }
    
    # 记录失败项
    for test, traceback_info in result.failures:
        report["failures"].append({
            "test": str(test),
            "traceback": traceback_info
        })
    
    # 记录错误项
    for test, traceback_info in result.errors:
        report["errors"].append({
            "test": str(test),
            "traceback": traceback_info
        })
    
    # 保存报告到文件
    report_path = os.path.join(project_root, 'logs', f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"测试报告已保存到: {report_path}")
    
    # 打印汇总信息
    logger.info("==== 测试汇总 ====")
    logger.info(f"总测试: {report['summary']['total']}")
    logger.info(f"成功: {report['summary']['success']}")
    logger.info(f"失败: {report['summary']['failure']}")
    logger.info(f"错误: {report['summary']['error']}")
    logger.info(f"运行时间: {report['summary']['runtime']}")
    logger.info(f"成功率: {report['summary']['success_rate']}")
    
    # 返回是否全部通过
    return len(result.failures) == 0 and len(result.errors) == 0


def run_all_tests():
    """运行所有测试用例"""
    logger.info("开始运行超神系统功能测试套件...")
    
    try:
        # 发现测试
        test_suite = discover_tests()
        
        # 运行测试
        logger.info("开始执行测试...")
        runner = unittest.TextTestRunner(verbosity=2)
        
        start_time = time.time()
        result = runner.run(test_suite)
        end_time = time.time()
        
        # 生成报告
        success = generate_report(result, start_time, end_time)
        
        # 返回是否成功
        return success
    
    except Exception as e:
        logger.error(f"运行测试时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 
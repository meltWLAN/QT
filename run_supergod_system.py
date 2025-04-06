#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神系统 - 中国股市量子预测引擎
启动入口文件
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import traceback

# 确保能够导入SuperQuantumNetwork包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SuperQuantumNetwork.china_market.controllers.market_controller import ChinaMarketController

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='超神系统 - 中国股市量子预测引擎')
    
    parser.add_argument('--config', '-c', type=str, 
                      help='配置文件路径')
    parser.add_argument('--stocks', '-s', type=str, 
                      help='关注的股票列表，以逗号分隔')
    parser.add_argument('--policy', '-p', action='store_true',
                      help='添加政策事件')
    parser.add_argument('--policy-type', type=str, 
                      help='政策类型')
    parser.add_argument('--policy-strength', type=float, default=0.5,
                      help='政策强度(0.0-1.0)')
    parser.add_argument('--policy-sectors', type=str, 
                      help='受影响的行业，以逗号分隔')
    parser.add_argument('--policy-desc', type=str, 
                      help='政策描述')
    
    return parser.parse_args()

def print_banner():
    """打印超神系统横幅"""
    banner = """
    ███████╗██╗   ██╗██████╗ ███████╗██████╗  ██████╗  ██████╗ ██████╗ 
    ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██╔════╝ ██╔═══██╗██╔══██╗
    ███████╗██║   ██║██████╔╝█████╗  ██████╔╝██║  ███╗██║   ██║██║  ██║
    ╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██║   ██║██║   ██║██║  ██║
    ███████║╚██████╔╝██║     ███████╗██║  ██║╚██████╔╝╚██████╔╝██████╔╝
    ╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═════╝ 
                                                                       
              量子纠缠预测引擎 v1.0 - 中国股市超神版
    ==================================================================
    """
    print(banner)

def initialize_logging():
    """初始化日志配置"""
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件路径
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'supergod_{timestamp}.log')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('supergod')

def main():
    """主函数"""
    # 打印横幅
    print_banner()
    
    # 初始化日志
    logger = initialize_logging()
    logger.info("超神系统启动中...")
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 确保需要的目录结构存在
    os.makedirs('cache/china_market/results', exist_ok=True)
    
    try:
        # 初始化控制器
        controller = ChinaMarketController(config_path=args.config)
        
        # 如果提供了股票列表，则更新配置
        if args.stocks:
            stocks = [s.strip() for s in args.stocks.split(',')]
            controller.config['default_stocks'] = stocks
            logger.info(f"更新关注股票: {stocks}")
        
        # 如果指定了添加政策事件
        if args.policy and args.policy_type and args.policy_sectors:
            sectors = [s.strip() for s in args.policy_sectors.split(',')]
            policy_desc = args.policy_desc or f"政策事件: {args.policy_type}"
            
            policy_id = controller.add_policy_event(
                policy_type=args.policy_type,
                strength=args.policy_strength,
                affected_sectors=sectors,
                description=policy_desc
            )
            
            if policy_id:
                logger.info(f"成功添加政策事件: {policy_id}")
            else:
                logger.error("添加政策事件失败")
        
        # 运行超神系统
        success = controller.run()
        
        if success:
            logger.info("超神系统运行完成！")
            
            # 输出股票推荐
            recommendations = controller.get_stock_recommendation()
            
            print("\n个股推荐详细信息：")
            print("="*70)
            print(f"{'代码':<8}{'名称':<12}{'行业':<12}{'操作':<10}{'现价':<8}{'目标价':<8}{'风险':<10}")
            print("-"*70)
            
            for rec in recommendations.get('recommendations', []):
                # 处理可能的None值
                stock = rec.get('stock', '')
                name = rec.get('name', '')
                sector = rec.get('sector', '')
                action = rec.get('action', '')
                current_price = rec.get('current_price', 0)
                limit_price = rec.get('limit_price')
                risk_level = rec.get('risk_level', '')
                
                # 格式化显示
                if limit_price is not None:
                    limit_price_display = f"{limit_price:<8.2f}"
                else:
                    limit_price_display = "---     "
                    
                print(f"{stock:<8}{name:<12}{sector:<12}{action:<10}{current_price:<8.2f}{limit_price_display}{risk_level:<10}")
            
            print("="*70)
            
            # 保存推荐结果
            output_dir = os.path.join(controller.config.get('cache_dir', './cache/china_market'), 'results')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with open(os.path.join(output_dir, f'recommendation_{timestamp}.json'), 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, ensure_ascii=False, indent=2)
        else:
            logger.error("超神系统运行失败")
            
    except Exception as e:
        logger.error(f"超神系统发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        print("\n运行出错，请查看日志文件获取详细信息。")
    
    print("\n感谢使用超神系统！")

if __name__ == "__main__":
    main() 
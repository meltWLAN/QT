"""
Web API接口 - 为量子共生网络提供HTTP API
允许外部系统获取交易决策和系统状态
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uvicorn

# 导入量子共生网络
from quantum_symbiotic_network.network import QuantumSymbioticNetwork
from quantum_symbiotic_network.data_sources import TushareDataSource
from quantum_symbiotic_network.simulation import MarketSimulator
from quantum_symbiotic_network.visualization.advanced_charts import AdvancedChartingEngine

# 设置日志
logger = logging.getLogger("quantum_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# API数据模型
class MarketData(BaseModel):
    """市场数据"""
    stocks: Dict[str, Dict[str, Any]] = Field(..., description="股票数据")
    indices: Dict[str, Dict[str, Any]] = Field(None, description="指数数据")
    timestamp: Optional[Any] = Field(None, description="时间戳")

class TradeDecision(BaseModel):
    """交易决策"""
    action: str = Field(..., description="交易动作: buy, sell, hold")
    confidence: float = Field(..., description="决策置信度")
    position_sizing: Optional[Dict[str, float]] = Field(None, description="仓位大小")
    timestamp: Optional[Any] = Field(None, description="时间戳")
    
class FeedbackData(BaseModel):
    """反馈数据"""
    performance: float = Field(..., description="表现评分")
    trade_result: str = Field(..., description="交易结果: success, failed")
    metrics: Optional[Dict[str, Any]] = Field(None, description="性能指标")
    
class SystemConfig(BaseModel):
    """系统配置"""
    network_config: Dict[str, Any] = Field(None, description="网络配置")
    data_source_config: Dict[str, Any] = Field(None, description="数据源配置")
    risk_config: Dict[str, Any] = Field(None, description="风险管理配置")
    visualization_config: Dict[str, Any] = Field(None, description="可视化配置")
    
class SystemStatus(BaseModel):
    """系统状态"""
    status: str = Field(..., description="系统状态")
    uptime: float = Field(..., description="运行时间(秒)")
    initialized: bool = Field(..., description="是否已初始化")
    performance_metrics: Dict[str, Any] = Field(None, description="性能指标")
    

# 创建API应用
app = FastAPI(
    title="量子共生网络交易系统 API",
    description="提供量子共生网络交易系统的Web API接口",
    version="1.0.0"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
quantum_network = None
data_source = None
market_simulator = None
charting_engine = None
start_time = time.time()
output_dir = os.path.join(os.path.dirname(__file__), "../../output")
decision_history = []

# 初始化函数
def initialize_system(config: Dict[str, Any] = None):
    """初始化系统组件"""
    global quantum_network, data_source, market_simulator, charting_engine
    
    if config is None:
        config = {}
        
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化量子共生网络
    network_config = config.get("network_config", {})
    quantum_network = QuantumSymbioticNetwork(network_config)
    
    # 初始化图表引擎
    vis_config = config.get("visualization_config", {})
    charting_engine = AdvancedChartingEngine(vis_config)
    
    # 初始化市场模拟器
    simulator_config = config.get("simulator_config", {
        "initial_capital": 1000000,
        "transaction_fee_rate": 0.0003,
        "slippage_rate": 0.0001,
        "risk_free_rate": 0.03 / 252
    })
    market_simulator = MarketSimulator(config=simulator_config)
    
    # 初始化数据源
    data_config = config.get("data_source_config", {})
    token = data_config.get("token", "")
    
    try:
        data_source = TushareDataSource(token=token)
    except Exception as e:
        logger.error(f"数据源初始化失败: {e}")
        data_source = None
        
    return {
        "status": "initialized",
        "network": quantum_network is not None,
        "data_source": data_source is not None,
        "market_simulator": market_simulator is not None,
        "charting_engine": charting_engine is not None
    }


# 依赖项 - 确保系统已初始化
async def get_network():
    """获取量子共生网络实例"""
    if quantum_network is None:
        raise HTTPException(status_code=500, detail="系统未初始化")
    return quantum_network


# API路由
@app.get("/", tags=["基础"])
async def root():
    """API根路径"""
    return {
        "name": "量子共生网络交易系统",
        "version": "1.0.0",
        "status": "running",
        "uptime": time.time() - start_time
    }

@app.post("/init", tags=["系统管理"], response_model=Dict[str, Any])
async def init_system(config: SystemConfig = None):
    """初始化系统"""
    config_dict = config.dict() if config else {}
    result = initialize_system(config_dict)
    return result

@app.get("/status", tags=["系统管理"], response_model=SystemStatus)
async def get_system_status(network: QuantumSymbioticNetwork = Depends(get_network)):
    """获取系统状态"""
    performance_metrics = network.get_performance_metrics()
    
    return {
        "status": "running",
        "uptime": time.time() - start_time,
        "initialized": True,
        "performance_metrics": performance_metrics
    }

@app.post("/decision", tags=["交易"], response_model=TradeDecision)
async def get_trade_decision(
    market_data: MarketData,
    network: QuantumSymbioticNetwork = Depends(get_network)
):
    """获取交易决策"""
    try:
        # 获取决策
        decision = network.step(market_data.dict())
        
        # 记录决策历史
        decision_history.append(decision)
        
        return decision
    except Exception as e:
        logger.error(f"获取决策失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取决策失败: {str(e)}")

@app.post("/feedback", tags=["交易"])
async def provide_feedback(
    feedback: FeedbackData,
    network: QuantumSymbioticNetwork = Depends(get_network)
):
    """提供反馈"""
    try:
        network.provide_feedback(feedback.dict())
        return {"status": "success", "message": "反馈已处理"}
    except Exception as e:
        logger.error(f"处理反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理反馈失败: {str(e)}")

@app.get("/market-data", tags=["数据"])
async def get_market_data(symbols: str = Query(None), days: int = 30):
    """获取市场数据"""
    if data_source is None:
        raise HTTPException(status_code=500, detail="数据源未初始化")
        
    try:
        # 解析股票代码
        if symbols:
            symbol_list = symbols.split(",")
        else:
            symbol_list = []
        
        # 获取日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        # 获取市场数据
        market_data = data_source.get_market_data(
            start_date=start_date,
            end_date=end_date,
            symbols=symbol_list
        )
        
        return market_data
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取市场数据失败: {str(e)}")

@app.get("/performance", tags=["性能分析"])
async def get_performance():
    """获取性能数据"""
    if market_simulator is None:
        raise HTTPException(status_code=500, detail="市场模拟器未初始化")
        
    try:
        performance = market_simulator.calculate_performance()
        return performance
    except Exception as e:
        logger.error(f"获取性能数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能数据失败: {str(e)}")

@app.get("/charts/{chart_type}", tags=["可视化"])
async def get_chart(chart_type: str, save: bool = False):
    """获取图表"""
    if charting_engine is None:
        raise HTTPException(status_code=500, detail="图表引擎未初始化")
        
    if market_simulator is None:
        raise HTTPException(status_code=500, detail="市场模拟器未初始化")
        
    try:
        # 获取性能数据
        performance = market_simulator.calculate_performance()
        
        # 根据图表类型生成不同图表
        save_path = None
        if save:
            save_path = os.path.join(output_dir, f"{chart_type}_chart.png")
            
        if chart_type == "performance":
            img_data = charting_engine.plot_performance_with_decisions(
                performance, decision_history, save_path
            )
        elif chart_type == "risk":
            # 简单风险数据
            risk_data = {
                "portfolio_risk": 0.12,
                "risk_status": "normal",
                "diversification_score": 0.65,
                "risk_contribution": {
                    "000001.SZ": 0.2,
                    "000002.SZ": 0.15,
                    "000063.SZ": 0.25,
                    "600036.SH": 0.4
                }
            }
            img_data = charting_engine.plot_risk_contribution(risk_data, save_path)
        elif chart_type == "correlation":
            # 示例相关性矩阵
            symbols = ["000001.SZ", "000002.SZ", "000063.SZ", "600036.SH"]
            corr_data = np.random.rand(len(symbols), len(symbols))
            # 确保对称矩阵
            corr_data = (corr_data + corr_data.T) / 2
            np.fill_diagonal(corr_data, 1)
            corr_df = pd.DataFrame(corr_data, index=symbols, columns=symbols)
            img_data = charting_engine.plot_correlation_heatmap(corr_df, save_path)
        elif chart_type == "quantum":
            # 示例量子状态数据
            quantum_states = {
                "000001.SZ": {
                    "amplitudes": np.random.rand(3),
                    "phases": np.random.rand(3)
                }
            }
            img_data = charting_engine.plot_quantum_decision_process(
                quantum_states, decision_history, save_path
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的图表类型: {chart_type}")
            
        if save and img_data:
            return {"status": "success", "file_path": save_path}
        elif img_data:
            # 返回Base64图片
            return {"status": "success", "image_data": img_data}
        else:
            raise HTTPException(status_code=500, detail="生成图表失败")
            
    except Exception as e:
        logger.error(f"生成图表失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成图表失败: {str(e)}")

@app.get("/charts/{chart_type}/file", tags=["可视化"])
async def get_chart_file(chart_type: str, background_tasks: BackgroundTasks):
    """获取图表文件"""
    if charting_engine is None:
        raise HTTPException(status_code=500, detail="图表引擎未初始化")
        
    try:
        # 获取性能数据
        if market_simulator is None:
            raise HTTPException(status_code=500, detail="市场模拟器未初始化")
            
        performance = market_simulator.calculate_performance()
        
        # 保存图表到临时文件
        save_path = os.path.join(output_dir, f"{chart_type}_chart.png")
        
        # 根据图表类型生成不同图表
        if chart_type == "performance":
            charting_engine.plot_performance_with_decisions(
                performance, decision_history, save_path
            )
        elif chart_type == "risk":
            # 简单风险数据
            risk_data = {
                "portfolio_risk": 0.12,
                "risk_status": "normal",
                "diversification_score": 0.65,
                "risk_contribution": {
                    "000001.SZ": 0.2,
                    "000002.SZ": 0.15,
                    "000063.SZ": 0.25,
                    "600036.SH": 0.4
                }
            }
            charting_engine.plot_risk_contribution(risk_data, save_path)
        elif chart_type == "correlation":
            # 示例相关性矩阵
            symbols = ["000001.SZ", "000002.SZ", "000063.SZ", "600036.SH"]
            corr_data = np.random.rand(len(symbols), len(symbols))
            # 确保对称矩阵
            corr_data = (corr_data + corr_data.T) / 2
            np.fill_diagonal(corr_data, 1)
            corr_df = pd.DataFrame(corr_data, index=symbols, columns=symbols)
            charting_engine.plot_correlation_heatmap(corr_df, save_path)
        elif chart_type == "quantum":
            # 示例量子状态数据
            quantum_states = {
                "000001.SZ": {
                    "amplitudes": np.random.rand(3),
                    "phases": np.random.rand(3)
                }
            }
            charting_engine.plot_quantum_decision_process(
                quantum_states, decision_history, save_path
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的图表类型: {chart_type}")
            
        # 设置任务在响应后删除文件
        def remove_file(path: str):
            try:
                time.sleep(30)  # 30秒后删除
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")
                
        background_tasks.add_task(remove_file, save_path)
        
        return FileResponse(
            path=save_path,
            filename=f"{chart_type}_chart.png",
            media_type="image/png"
        )
            
    except Exception as e:
        logger.error(f"生成图表失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成图表失败: {str(e)}")

@app.post("/save-model", tags=["系统管理"])
async def save_model(path: str = None, network: QuantumSymbioticNetwork = Depends(get_network)):
    """保存模型"""
    try:
        if path is None:
            path = os.path.join(output_dir, "model")
            
        network.save(path)
        return {"status": "success", "message": f"模型已保存至 {path}"}
    except Exception as e:
        logger.error(f"保存模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存模型失败: {str(e)}")

@app.post("/load-model", tags=["系统管理"])
async def load_model(path: str = None, network: QuantumSymbioticNetwork = Depends(get_network)):
    """加载模型"""
    try:
        if path is None:
            path = os.path.join(output_dir, "model")
            
        network.load(path)
        return {"status": "success", "message": f"从 {path} 加载模型成功"}
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


# 启动服务器函数
def run_server(host="0.0.0.0", port=8000, init=True):
    """启动API服务器"""
    if init:
        # 自动初始化系统
        initialize_system()
        
    # 启动FastAPI服务器
    uvicorn.run(app, host=host, port=port)
    

if __name__ == "__main__":
    # 当作为脚本直接运行时执行
    run_server() 
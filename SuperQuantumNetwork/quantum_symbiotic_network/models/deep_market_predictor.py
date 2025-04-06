#!/usr/bin/env python3
"""
深度市场预测器 - 基于深度学习的市场分析与预测模型
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
import time
from collections import deque
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepMarketPredictor")

# 尝试导入机器学习库
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    logger.info("TensorFlow已加载")
    
    # 限制GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"限制GPU内存增长：检测到 {len(gpus)} 个GPU设备")
        except RuntimeError as e:
            logger.warning(f"GPU内存设置失败: {e}")
except ImportError:
    logger.warning("TensorFlow未安装，深度学习功能将不可用。请使用pip install tensorflow安装。")
    TF_AVAILABLE = False

class DeepMarketPredictor:
    """深度学习市场预测模型"""
    
    def __init__(self, config=None):
        """初始化深度市场预测器
        
        Args:
            config (dict): 配置信息
        """
        self.config = config or {}
        self.model_dir = os.path.join("quantum_symbiotic_network", "models", "saved_models")
        self.model_path = os.path.join(self.model_dir, "deep_market_predictor.h5")
        self.history_path = os.path.join(self.model_dir, "training_history.pkl")
        self.scaler_path = os.path.join(self.model_dir, "feature_scaler.pkl")
        
        # 创建模型保存目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 模型参数
        self.sequence_length = self.config.get("sequence_length", 20)  # 时序长度
        self.prediction_horizon = self.config.get("prediction_horizon", 5)  # 预测长度
        self.batch_size = self.config.get("batch_size", 64)
        self.epochs = self.config.get("epochs", 50)
        self.dropout_rate = self.config.get("dropout_rate", 0.2)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.validation_split = self.config.get("validation_split", 0.2)
        
        # 特征缩放器
        self.scaler = None
        self.model = None
        
        # 模型预热记忆
        self.warmup_memory = deque(maxlen=self.sequence_length)
        
        # 初始化模型
        if TF_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """初始化深度学习模型"""
        try:
            # 尝试加载现有模型
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                logger.info(f"已加载预训练模型：{self.model_path}")
                
                # 加载特征缩放器
                if os.path.exists(self.scaler_path):
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    logger.info("已加载特征缩放器")
            else:
                logger.info("未找到预训练模型，将在训练时创建新模型")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model = None
    
    def _build_hybrid_model(self, input_shape, num_macro_features):
        """构建混合深度学习模型（CNN-LSTM架构）
        
        Args:
            input_shape (tuple): 输入数据形状
            num_macro_features (int): 宏观特征数量
            
        Returns:
            Model: Keras模型
        """
        # 技术指标输入
        tech_input = Input(shape=input_shape, name='technical_input')
        
        # CNN部分
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(tech_input)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Dropout(self.dropout_rate)(cnn)
        
        # LSTM部分
        lstm = LSTM(units=128, return_sequences=True)(cnn)
        lstm = Dropout(self.dropout_rate)(lstm)
        lstm = LSTM(units=64)(lstm)
        lstm = Dropout(self.dropout_rate)(lstm)
        
        # 宏观特征输入
        if num_macro_features > 0:
            macro_input = Input(shape=(num_macro_features,), name='macro_input')
            macro_dense = Dense(32, activation='relu')(macro_input)
            macro_dense = Dropout(self.dropout_rate)(macro_dense)
            
            # 合并技术指标和宏观特征
            merged = Concatenate()([lstm, macro_dense])
            dense = Dense(64, activation='relu')(merged)
            
            # 多输出预测
            price_pred = Dense(self.prediction_horizon, name='price_prediction')(dense)
            volatility_pred = Dense(self.prediction_horizon, name='volatility_prediction')(dense)
            trend_pred = Dense(self.prediction_horizon, activation='tanh', name='trend_prediction')(dense)
            
            # 构建模型
            model = Model(inputs=[tech_input, macro_input], 
                          outputs=[price_pred, volatility_pred, trend_pred])
        else:
            # 没有宏观特征时的模型
            dense = Dense(64, activation='relu')(lstm)
            
            # 多输出预测
            price_pred = Dense(self.prediction_horizon, name='price_prediction')(dense)
            volatility_pred = Dense(self.prediction_horizon, name='volatility_prediction')(dense)
            trend_pred = Dense(self.prediction_horizon, activation='tanh', name='trend_prediction')(dense)
            
            # 构建模型
            model = Model(inputs=tech_input, 
                          outputs=[price_pred, volatility_pred, trend_pred])
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                'price_prediction': 'mse',
                'volatility_prediction': 'mse',
                'trend_prediction': 'mse'
            },
            metrics={
                'price_prediction': ['mae'],
                'volatility_prediction': ['mae'],
                'trend_prediction': ['accuracy']
            }
        )
        
        return model
    
    def _preprocess_data(self, stock_data):
        """预处理股票数据
        
        Args:
            stock_data (DataFrame): 股票数据
            
        Returns:
            tuple: 预处理后的特征、目标值、宏观特征
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # 提取特征
        price_cols = ['open', 'high', 'low', 'close']
        tech_cols = ['ma5', 'ma10', 'ma20', 'rsi14', 'macd', 'signal', 'volume_ratio']
        advanced_cols = ['rsi_change', 'bb_width', 'momentum', 'adx', 'k_value', 'd_value', 'j_value']
        
        # 检查特征是否存在
        available_tech_cols = [col for col in tech_cols if col in stock_data.columns]
        available_advanced_cols = [col for col in advanced_cols if col in stock_data.columns]
        
        # 合并特征
        feature_cols = price_cols + available_tech_cols + available_advanced_cols
        
        # 添加波动率特征
        stock_data['volatility'] = stock_data['high'] - stock_data['low']
        stock_data['returns'] = stock_data['close'].pct_change()
        stock_data['direction'] = np.sign(stock_data['returns'])
        
        # 去除NaN
        stock_data = stock_data.dropna()
        
        # 提取宏观数据(如果有)
        macro_features = None
        if 'macro_data' in stock_data.columns:
            macro_features = stock_data['macro_data'].values
        
        # 特征缩放
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            features_scaled = self.scaler.fit_transform(stock_data[feature_cols])
            
            # 保存缩放器
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            features_scaled = self.scaler.transform(stock_data[feature_cols])
        
        # 准备序列数据
        X, y_price, y_volatility, y_trend = [], [], [], []
        
        for i in range(len(features_scaled) - self.sequence_length - self.prediction_horizon + 1):
            # 提取序列
            seq = features_scaled[i:i+self.sequence_length]
            X.append(seq)
            
            # 提取价格目标（使用收盘价）
            price_target = stock_data['close'].iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon].values
            price_current = stock_data['close'].iloc[i+self.sequence_length-1]
            price_target_normalized = (price_target - price_current) / price_current  # 相对收益率
            y_price.append(price_target_normalized)
            
            # 提取波动率目标
            volatility_target = stock_data['volatility'].iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon].values
            volatility_mean = stock_data['volatility'].iloc[i:i+self.sequence_length].mean()
            volatility_target_normalized = volatility_target / volatility_mean if volatility_mean > 0 else volatility_target
            y_volatility.append(volatility_target_normalized)
            
            # 提取趋势目标
            trend_target = stock_data['direction'].iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon].values
            y_trend.append(trend_target)
        
        return np.array(X), np.array(y_price), np.array(y_volatility), np.array(y_trend), macro_features
    
    def train(self, stock_data, macro_data=None):
        """训练模型
        
        Args:
            stock_data (DataFrame): 股票数据
            macro_data (DataFrame, optional): 宏观经济数据
            
        Returns:
            dict: 训练历史记录
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow未安装，无法训练模型")
            return None
        
        try:
            # 合并宏观数据
            if macro_data is not None:
                # 提取相关宏观特征
                macro_features = self._extract_macro_features(macro_data)
                # 将宏观特征添加到股票数据
                stock_data = self._merge_macro_features(stock_data, macro_features)
                num_macro_features = macro_features.shape[1] if len(macro_features.shape) > 1 else 1
            else:
                num_macro_features = 0
            
            # 预处理数据
            X, y_price, y_volatility, y_trend, macro_X = self._preprocess_data(stock_data)
            
            if len(X) == 0:
                logger.error("处理后数据为空，无法训练模型")
                return None
            
            logger.info(f"预处理后数据：{X.shape}, 价格预测目标：{y_price.shape}, 波动率目标：{y_volatility.shape}, 趋势目标：{y_trend.shape}")
            
            # 构建模型
            input_shape = (self.sequence_length, X.shape[2])
            if self.model is None:
                self.model = self._build_hybrid_model(input_shape, num_macro_features)
                logger.info(f"创建新模型，输入形状：{input_shape}")
            
            # 设置回调
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(self.model_path, save_best_only=True)
            ]
            
            # 训练模型
            if num_macro_features > 0 and macro_X is not None:
                history = self.model.fit(
                    [X, macro_X], 
                    [y_price, y_volatility, y_trend],
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history = self.model.fit(
                    X, 
                    [y_price, y_volatility, y_trend],
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=callbacks,
                    verbose=1
                )
            
            # 保存训练历史
            with open(self.history_path, 'wb') as f:
                pickle.dump(history.history, f)
            
            logger.info("模型训练完成，已保存模型和训练历史")
            return history.history
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return None
    
    def _extract_macro_features(self, macro_data):
        """从宏观数据中提取特征
        
        Args:
            macro_data (dict): 宏观经济数据
            
        Returns:
            ndarray: 宏观特征
        """
        features = []
        
        # 处理货币供应量数据
        if 'money_supply' in macro_data and not macro_data['money_supply'].empty:
            ms = macro_data['money_supply']
            # 提取最新的一行
            latest_ms = ms.iloc[-1]
            features.extend([
                latest_ms.get('m1_yoy', 0),
                latest_ms.get('m2_yoy', 0)
            ])
        
        # 处理GDP数据
        if 'gdp' in macro_data and not macro_data['gdp'].empty:
            gdp = macro_data['gdp']
            latest_gdp = gdp.iloc[-1]
            features.extend([
                latest_gdp.get('gdp_yoy', 0),
                latest_gdp.get('pi_yoy', 0),
                latest_gdp.get('si_yoy', 0),
                latest_gdp.get('ti_yoy', 0)
            ])
        
        # 处理CPI数据
        if 'cpi' in macro_data and not macro_data['cpi'].empty:
            cpi = macro_data['cpi']
            latest_cpi = cpi.iloc[-1]
            features.append(latest_cpi.get('nt_yoy', 0))
        
        # 处理PPI数据
        if 'ppi' in macro_data and not macro_data['ppi'].empty:
            ppi = macro_data['ppi']
            latest_ppi = ppi.iloc[-1]
            features.append(latest_ppi.get('nt_yoy', 0))
        
        return np.array(features)
    
    def _merge_macro_features(self, stock_data, macro_features):
        """将宏观特征合并到股票数据
        
        Args:
            stock_data (DataFrame): 股票数据
            macro_features (ndarray): 宏观特征
            
        Returns:
            DataFrame: 合并后的数据
        """
        # 创建一个全部是宏观特征的列
        stock_data['macro_data'] = [macro_features] * len(stock_data)
        return stock_data
    
    def predict(self, stock_data, macro_data=None):
        """进行预测
        
        Args:
            stock_data (DataFrame): 股票数据
            macro_data (DataFrame, optional): 宏观经济数据
            
        Returns:
            dict: 预测结果，包含价格变化预测、波动率预测和趋势预测
        """
        if not TF_AVAILABLE or self.model is None:
            logger.error("模型未加载或TensorFlow未安装，无法进行预测")
            return None
        
        try:
            # 检查数据长度是否足够
            if len(stock_data) < self.sequence_length:
                logger.warning(f"数据长度不足 ({len(stock_data)} < {self.sequence_length})，无法进行预测")
                return None
            
            # 合并宏观数据
            has_macro = False
            if macro_data is not None:
                # 提取相关宏观特征
                macro_features = self._extract_macro_features(macro_data)
                # 将宏观特征添加到股票数据
                stock_data = self._merge_macro_features(stock_data, macro_features)
                has_macro = True
            
            # 提取需要的特征
            price_cols = ['open', 'high', 'low', 'close']
            tech_cols = ['ma5', 'ma10', 'ma20', 'rsi14', 'macd', 'signal', 'volume_ratio']
            advanced_cols = ['rsi_change', 'bb_width', 'momentum', 'adx', 'k_value', 'd_value', 'j_value']
            
            # 检查特征是否存在
            available_tech_cols = [col for col in tech_cols if col in stock_data.columns]
            available_advanced_cols = [col for col in advanced_cols if col in stock_data.columns]
            
            # 合并特征
            feature_cols = price_cols + available_tech_cols + available_advanced_cols
            
            # 确保所有必要的特征都存在
            missing_cols = [col for col in price_cols if col not in stock_data.columns]
            if missing_cols:
                logger.error(f"缺少必要的价格特征: {missing_cols}")
                return None
            
            # 缩放特征
            features = stock_data[feature_cols].values[-self.sequence_length:]
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                logger.error("特征缩放器未加载，无法缩放特征")
                return None
            
            # 准备输入数据
            X = np.array([features_scaled])
            
            # 提取宏观特征(如果有)
            if has_macro and 'macro_data' in stock_data.columns:
                macro_X = np.array([stock_data['macro_data'].iloc[-1]])
                predictions = self.model.predict([X, macro_X])
            else:
                predictions = self.model.predict(X)
            
            # 当前收盘价
            current_price = stock_data['close'].iloc[-1]
            
            # 提取预测结果
            price_pred, volatility_pred, trend_pred = predictions
            
            # 转换价格预测为绝对价格
            absolute_price_pred = current_price * (1 + price_pred[0])
            
            # 计算置信度
            confidence = self._calculate_prediction_confidence(price_pred[0], trend_pred[0])
            
            # 返回预测结果
            result = {
                'price_prediction': absolute_price_pred.tolist(),
                'price_change_percent': price_pred[0].tolist(),
                'volatility_prediction': volatility_pred[0].tolist(),
                'trend_prediction': trend_pred[0].tolist(),
                'confidence': confidence,
                'prediction_horizon': self.prediction_horizon,
                'current_price': float(current_price),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None
    
    def _calculate_prediction_confidence(self, price_pred, trend_pred):
        """计算预测置信度
        
        Args:
            price_pred (ndarray): 价格变化预测
            trend_pred (ndarray): 趋势预测
            
        Returns:
            float: 置信度（0-1之间）
        """
        # 价格预测的大小表示变化强度，可以作为置信度的一部分
        price_change_strength = np.abs(price_pred)
        
        # 趋势预测的绝对值接近1表示更确定的方向
        trend_certainty = np.abs(trend_pred)
        
        # 价格预测和趋势预测符号一致表示更高的置信度
        direction_consistency = np.where(np.sign(price_pred) == np.sign(trend_pred), 1.0, 0.5)
        
        # 综合计算置信度
        combined_confidence = 0.4 * price_change_strength + 0.4 * trend_certainty + 0.2 * direction_consistency
        
        # 将置信度值缩放到0-1之间
        scaled_confidence = np.clip(combined_confidence, 0, 1)
        
        return scaled_confidence.tolist()
    
    def evaluate(self, stock_data, target_data, macro_data=None):
        """评估模型性能
        
        Args:
            stock_data (DataFrame): 股票数据
            target_data (DataFrame): 实际结果数据
            macro_data (DataFrame, optional): 宏观经济数据
            
        Returns:
            dict: 评估结果
        """
        if not TF_AVAILABLE or self.model is None:
            logger.error("模型未加载或TensorFlow未安装，无法评估模型")
            return None
        
        try:
            # 获取预测结果
            predictions = self.predict(stock_data, macro_data)
            if predictions is None:
                return None
            
            # 提取预测值
            price_pred = predictions['price_prediction']
            trend_pred = predictions['trend_prediction']
            
            # 提取实际值
            actual_prices = target_data['close'].values[:self.prediction_horizon]
            actual_returns = target_data['close'].pct_change().values[:self.prediction_horizon]
            actual_trends = np.sign(actual_returns)
            
            # 计算价格预测误差
            price_errors = np.abs(np.array(price_pred) - actual_prices)
            
            # 计算方向准确率
            predicted_trends = np.sign(np.diff(np.array([stock_data['close'].iloc[-1]] + price_pred)))
            direction_accuracy = np.mean(predicted_trends == actual_trends[1:])
            
            # 计算均方根误差
            rmse = np.sqrt(np.mean(np.square(price_errors)))
            
            # 计算平均绝对误差
            mae = np.mean(price_errors)
            
            # 计算平均绝对百分比误差
            mape = np.mean(np.abs(price_errors / actual_prices)) * 100
            
            # 返回评估结果
            return {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy)
            }
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return None
    
    def load_model(self, model_path=None):
        """加载预训练模型
        
        Args:
            model_path (str, optional): 模型路径，默认使用初始化时设定的路径
            
        Returns:
            bool: 是否成功加载模型
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow未安装，无法加载模型")
            return False
        
        try:
            # 如果提供了新路径，则使用新路径
            if model_path is not None:
                self.model_path = model_path
            
            # 加载模型
            self.model = load_model(self.model_path)
            logger.info(f"成功加载模型：{self.model_path}")
            
            # 尝试加载特征缩放器
            scaler_path = os.path.join(os.path.dirname(self.model_path), "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("成功加载特征缩放器")
            
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def save_model(self, model_path=None):
        """保存模型
        
        Args:
            model_path (str, optional): 模型保存路径，默认使用初始化时设定的路径
            
        Returns:
            bool: 是否成功保存模型
        """
        if not TF_AVAILABLE or self.model is None:
            logger.error("模型未加载或TensorFlow未安装，无法保存模型")
            return False
        
        try:
            # 如果提供了新路径，则使用新路径
            if model_path is not None:
                self.model_path = model_path
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # 保存模型
            self.model.save(self.model_path)
            logger.info(f"模型已保存到：{self.model_path}")
            
            # 保存特征缩放器
            if self.scaler is not None:
                scaler_path = os.path.join(os.path.dirname(self.model_path), "feature_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info(f"特征缩放器已保存到：{scaler_path}")
            
            return True
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return False 
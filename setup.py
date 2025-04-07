#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
超神系统中国市场分析模块安装脚本
"""

import os
from setuptools import setup, find_packages

# 读取README文件
with open(os.path.join(os.path.dirname(__file__), 'SuperQuantumNetwork/README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

# 安装配置
setup(
    name='SuperQuantumNetwork',
    version='1.0.0',
    description='超神系统中国市场分析模块',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='超神系统开发团队',
    author_email='contact@superquantum.io',
    url='https://github.com/supergod-quantum/market-module',
    packages=find_packages(),
    install_requires=[
        'PyQt5>=5.15.0',
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'requests>=2.25.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    python_requires='>=3.7',
    keywords='finance, market analysis, trading, quantum AI',
    project_urls={
        'Bug Reports': 'https://github.com/supergod-quantum/market-module/issues',
        'Source': 'https://github.com/supergod-quantum/market-module',
    },
) 
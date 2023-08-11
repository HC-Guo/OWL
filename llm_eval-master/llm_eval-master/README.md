# 安装
```pip install -r requirements.txt```

当评测多选问题集需要使用bleurt指标时，需要额外安装[bleurt](https://github.com/google-research/bleurt#installation) 模块，安装方法如下：
```
pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

# 使用
```python ./web_tool.py```直接运行web_tool.py文件，然后在浏览器中根据提示进行操作即可。
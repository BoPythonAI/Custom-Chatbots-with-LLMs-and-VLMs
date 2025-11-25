#!/bin/bash
# 启动Streamlit Web界面

cd /root/autodl-tmp/SQA

# 激活虚拟环境
if [ ! -d "venv" ]; then
    echo "虚拟环境不存在，请先运行 ./setup_env.sh"
    exit 1
fi

source venv/bin/activate

# 检查streamlit是否安装
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit未安装，正在安装..."
    pip install streamlit>=1.28.0
fi

# 检查端口8501是否被占用
PORT=8501
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":$PORT " || ss -tuln 2>/dev/null | grep -q ":$PORT "; then
    echo "⚠️  端口 $PORT 已被占用，正在停止旧进程..."
    pkill -f "streamlit run.*port $PORT" 2>/dev/null
    sleep 2
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  无法停止旧进程，尝试使用端口 8502..."
        PORT=8502
    else
        echo "✅ 旧进程已停止"
    fi
fi

# 启动Streamlit应用
echo "=========================================="
echo "启动Streamlit Web界面"
echo "=========================================="
echo ""
echo "访问地址: http://localhost:$PORT"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0


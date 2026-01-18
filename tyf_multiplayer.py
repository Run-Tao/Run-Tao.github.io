import cv2
import mediapipe as mp
import numpy as np
import random
import time
import requests
import base64
import json
from PIL import ImageFont, ImageDraw, Image
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import os
import socket

# 创建保存目录
save_dir = "drawings"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 游戏状态
class GameState:
    def __init__(self):
        # 游戏词库
        self.words = [
            "苹果", "香蕉", "猫", "狗", "房子", "汽车", "飞机", "船", "树", "花",
            "太阳", "月亮", "星星", "雨伞", "眼镜", "帽子", "鞋子", "衣服", "手机", "电脑",
            "电视", "冰箱", "洗衣机", "自行车", "摩托车", "火车", "火箭", "足球", "篮球", "乒乓球"
        ]
        self.current_word = self.get_random_word()
        self.canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self.drawing = False
        self.last_x, self.last_y = 0, 0
        self.ai_guess = ""
        self.hint = ""
        self.guesses = []  # 存储所有猜测
        self.is_game_active = True
        self.current_color = (0, 0, 0)  # 默认颜色：黑色 (BGR格式)
    
    def get_random_word(self):
        """获取随机词语"""
        return random.choice(self.words)
    
    def reset_game(self):
        """重置游戏"""
        self.current_word = self.get_random_word()
        self.canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self.drawing = False
        self.last_x, self.last_y = 0, 0
        self.ai_guess = ""
        self.hint = ""
        self.guesses = []
        self.is_game_active = True
    
    def add_guess(self, guess):
        """添加猜测"""
        self.guesses.append(guess)
        return self.check_guess(guess)
    
    def check_guess(self, guess):
        """检查猜测是否正确"""
        return guess == self.current_word
    
    def update_canvas(self, x, y, drawing, color=None):
        """更新画布，支持自定义颜色"""
        # 如果提供了颜色，更新当前颜色
        if color is not None:
            self.current_color = tuple(color)  # 转换为元组
        
        if drawing:
            if self.last_x != 0 and self.last_y != 0:
                # 使用当前颜色绘制线条
                cv2.line(self.canvas, (self.last_x, self.last_y), (x, y), self.current_color, 2)
            self.last_x, self.last_y = x, y
        else:
            self.last_x, self.last_y = 0, 0
    
    def clear_canvas(self):
        """清空画布"""
        self.canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

# 创建游戏状态实例
game_state = GameState()

# 创建FastAPI应用
app = FastAPI(title="双人你画我猜游戏")

# 添加CORS配置，允许跨域访问
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源访问，生产环境可指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)

# 创建WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.drawers: list[WebSocket] = []  # 画画的人
        self.guessers: list[WebSocket] = []  # 猜词的人
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.drawers:
            self.drawers.remove(websocket)
        if websocket in self.guessers:
            self.guessers.remove(websocket)
    
    async def broadcast(self, message: dict):
        """向所有连接的客户端广播消息"""
        for connection in self.active_connections:
            await connection.send_json(message)
    
    async def broadcast_to_guessers(self, message: dict):
        """向所有猜词的人广播消息"""
        for connection in self.guessers:
            await connection.send_json(message)
    
    async def broadcast_to_drawers(self, message: dict):
        """向所有画画的人广播消息"""
        for connection in self.drawers:
            await connection.send_json(message)
    
    def add_drawer(self, websocket: WebSocket):
        """添加画画的人"""
        if websocket not in self.drawers:
            self.drawers.append(websocket)
        if websocket in self.guessers:
            self.guessers.remove(websocket)
    
    def add_guesser(self, websocket: WebSocket):
        """添加猜词的人"""
        if websocket not in self.guessers:
            self.guessers.append(websocket)
        if websocket in self.drawers:
            self.drawers.remove(websocket)

# 创建连接管理器实例
manager = ConnectionManager()

# 处理WebSocket连接
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            # 处理不同类型的消息
            if data["type"] == "register":
                # 注册用户类型
                if data["role"] == "drawer":
                    manager.add_drawer(websocket)
                elif data["role"] == "guesser":
                    manager.add_guesser(websocket)
                await websocket.send_json({
                    "type": "game_state",
                    "current_word": game_state.current_word,
                    "is_game_active": game_state.is_game_active
                })
            
            elif data["type"] == "draw":
                # 更新画布
                x = data["x"]
                y = data["y"]
                drawing = data["drawing"]
                # 获取颜色信息，如果没有提供则使用当前颜色
                color = data.get("color", None)
                game_state.update_canvas(x, y, drawing, color)
                
                # 广播画布更新
                await manager.broadcast_to_guessers({
                    "type": "canvas_update",
                    "canvas": base64.b64encode(cv2.imencode('.jpg', game_state.canvas)[1]).decode('utf-8')
                })
            
            elif data["type"] == "clear":
                # 清空画布
                game_state.clear_canvas()
                await manager.broadcast_to_guessers({
                    "type": "canvas_update",
                    "canvas": base64.b64encode(cv2.imencode('.jpg', game_state.canvas)[1]).decode('utf-8')
                })
            
            elif data["type"] == "canvas_update":
                # 从客户端接收画布更新（当画画者按f键保存并上传时）
                canvas_data = data.get("canvas", None)
                if canvas_data:
                    # 直接广播客户端上传的画布数据给所有猜词者
                    await manager.broadcast_to_guessers({
                        "type": "canvas_update",
                        "canvas": canvas_data
                    })
            
            elif data["type"] == "guess":
                # 处理猜词
                guess = data["guess"]
                is_correct = game_state.add_guess(guess)
                
                # 向所有客户端广播猜测结果
                await manager.broadcast({
                    "type": "guess_result",
                    "guess": guess,
                    "is_correct": is_correct,
                    "guesses": game_state.guesses
                })
                
                if is_correct:
                    # 游戏结束，重置游戏
                    game_state.reset_game()
                    await manager.broadcast({
                        "type": "game_reset",
                        "current_word": game_state.current_word
                    })
            
            elif data["type"] == "reset":
                # 重置游戏
                game_state.reset_game()
                await manager.broadcast({
                    "type": "game_reset",
                    "current_word": game_state.current_word
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 创建静态文件目录
if not os.path.exists("static"):
    os.makedirs("static")

# 提供静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 保存HTML文件
html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>双人你画我猜游戏</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .role-selection {
            text-align: center;
            margin-bottom: 20px;
        }
        .role-selection button {
            padding: 10px 20px;
            margin: 0 10px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            background-color: #4CAF50;
            color: white;
        }
        .role-selection button:hover {
            background-color: #45a049;
        }
        .game-area {
            display: flex;
            gap: 20px;
        }
        .canvas-container {
            flex: 1;
        }
        canvas {
            border: 2px solid #333;
            border-radius: 5px;
            cursor: crosshair;
        }
        .controls {
            margin-top: 10px;
        }
        .controls button {
            padding: 8px 16px;
            margin-right: 10px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            background-color: #008CBA;
            color: white;
        }
        .controls button:hover {
            background-color: #007B9A;
        }
        .guess-area {
            width: 300px;
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
        }
        .guess-input {
            margin-bottom: 10px;
        }
        .guess-input input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .guess-input button {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border: none;
            border-radius: 5px;
            background-color: #f44336;
            color: white;
            cursor: pointer;
        }
        .guess-input button:hover {
            background-color: #d32f2f;
        }
        .guess-history {
            margin-top: 20px;
        }
        .guess-history h3 {
            margin-bottom: 10px;
        }
        .guess-item {
            margin-bottom: 5px;
            padding: 5px;
            border-radius: 3px;
        }
        .correct {
            background-color: #d4edda;
            color: #155724;
        }
        .incorrect {
            background-color: #f8d7da;
            color: #721c24;
        }
        .word-display {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>双人你画我猜游戏</h1>
        
        <!-- 角色选择 -->
        <div class="role-selection" id="roleSelection">
            <h2>请选择你的角色</h2>
            <button onclick="registerRole('drawer')">我要画画</button>
            <button onclick="registerRole('guesser')">我要猜词</button>
        </div>
        
        <!-- 画画区域 -->
        <div class="game-area hidden" id="drawerArea">
            <div class="canvas-container">
                <div class="word-display">
                    <h3>你需要画：<span id="currentWord"></span></h3>
                </div>
                <canvas id="drawingCanvas" width="640" height="480"></canvas>
                <div class="controls">
                    <button onclick="clearCanvas()">清空画布</button>
                    <button onclick="resetGame()">重新开始</button>
                </div>
            </div>
            <div class="guess-area">
                <h3>猜测记录</h3>
                <div id="guessHistory"></div>
            </div>
        </div>
        
        <!-- 猜词区域 -->
        <div class="game-area hidden" id="guesserArea">
            <div class="canvas-container">
                <h3>请猜画的是什么</h3>
                <canvas id="viewCanvas" width="640" height="480"></canvas>
            </div>
            <div class="guess-area">
                <div class="guess-input">
                    <input type="text" id="guessInput" placeholder="请输入你的猜测...">
                    <button onclick="submitGuess()">提交猜测</button>
                </div>
                <div class="guess-history">
                    <h3>猜测记录</h3>
                    <div id="guessHistory2"></div>
                </div>
                <!-- 正确答案显示区域 -->
                <div id="correctAnswer" style="margin-top: 20px; padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px; display: none;"></div>
                <button onclick="resetGame()" style="margin-top: 20px;">重新开始</button>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let role = '';
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        
        // 初始化WebSocket连接
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            ws = new WebSocket(`${protocol}//${host}/ws`);
            
            ws.onopen = function() {
                console.log('WebSocket连接已建立');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket连接已关闭');
                setTimeout(initWebSocket, 1000); // 尝试重新连接
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket错误:', error);
            };
        }
        
        // 处理接收到的消息
        function handleMessage(data) {
            if (data.type === 'game_state') {
                document.getElementById('currentWord').textContent = data.current_word;
            } else if (data.type === 'canvas_update') {
                // 更新画布
                const img = new Image();
                img.onload = function() {
                    const canvas = document.getElementById('viewCanvas');
                    const ctx = canvas.getContext('2d');
                    // 确保画布背景为白色，支持彩色显示
                    ctx.fillStyle = 'white';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    // 绘制彩色图像
                    ctx.drawImage(img, 0, 0);
                };
                img.src = 'data:image/jpeg;base64,' + data.canvas;
            } else if (data.type === 'guess_result') {
                // 更新猜测记录
                updateGuessHistory(data);
                
                // 如果猜测正确，显示正确答案
                if (data.is_correct) {
                    showCorrectAnswer();
                }
            } else if (data.type === 'game_reset') {
                // 重置游戏
                document.getElementById('currentWord').textContent = data.current_word;
                clearCanvas();
                document.getElementById('guessInput').value = '';
                document.getElementById('guessHistory').innerHTML = '';
                document.getElementById('guessHistory2').innerHTML = '';
                // 隐藏正确答案显示
                hideCorrectAnswer();
            }
        }
        
        // 注册用户角色
        function registerRole(roleType) {
            role = roleType;
            ws.send(JSON.stringify({ type: 'register', role: role }));
            
            // 显示相应的游戏区域
            document.getElementById('roleSelection').classList.add('hidden');
            if (roleType === 'drawer') {
                document.getElementById('drawerArea').classList.remove('hidden');
                initDrawingCanvas();
            } else {
                document.getElementById('guesserArea').classList.remove('hidden');
            }
        }
        
        // 初始化画画画布
        function initDrawingCanvas() {
            const canvas = document.getElementById('drawingCanvas');
            const ctx = canvas.getContext('2d');
            
            // 确保画布初始化为白色背景
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 设置画布样式
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.strokeStyle = 'black';
            ctx.fillStyle = 'black';
            
            // 鼠标事件
            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mouseout', stopDrawing);
            
            // 触摸事件
            canvas.addEventListener('touchstart', (e) => {
                e.preventDefault();
                const rect = canvas.getBoundingClientRect();
                const touch = e.touches[0];
                lastX = touch.clientX - rect.left;
                lastY = touch.clientY - rect.top;
                startDrawing(e);
            });
            canvas.addEventListener('touchmove', (e) => {
                e.preventDefault();
                draw(e);
            });
            canvas.addEventListener('touchend', stopDrawing);
        }
        
        // 开始绘制
        function startDrawing(e) {
            isDrawing = true;
            const rect = e.target.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            lastX = x;
            lastY = y;
        }
        
        // 绘制
        function draw(e) {
            if (!isDrawing) return;
            
            const rect = e.target.getBoundingClientRect();
            const x = e.type.includes('touch') ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
            const y = e.type.includes('touch') ? e.touches[0].clientY - rect.top : e.clientY - rect.top;
            
            // 绘制本地画布
            const ctx = document.getElementById('drawingCanvas').getContext('2d');
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            
            // 发送绘制数据到服务器
            ws.send(JSON.stringify({
                type: 'draw',
                x: x,
                y: y,
                drawing: true
            }));
            
            lastX = x;
            lastY = y;
        }
        
        // 停止绘制
        function stopDrawing() {
            if (isDrawing) {
                isDrawing = false;
                ws.send(JSON.stringify({
                    type: 'draw',
                    x: lastX,
                    y: lastY,
                    drawing: false
                }));
            }
        }
        
        // 清空画布
        function clearCanvas() {
            const ctx = document.getElementById('drawingCanvas').getContext('2d');
            // 使用白色填充整个画布，确保彩色支持
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, 640, 480);
            ws.send(JSON.stringify({ type: 'clear' }));
        }
        
        // 提交猜测
        function submitGuess() {
            const guess = document.getElementById('guessInput').value.trim();
            if (guess) {
                ws.send(JSON.stringify({ type: 'guess', guess: guess }));
                document.getElementById('guessInput').value = '';
            }
        }
        
        // 重置游戏
        function resetGame() {
            ws.send(JSON.stringify({ type: 'reset' }));
        }
        
        // 更新猜测记录
        function updateGuessHistory(data) {
            const guessItem = document.createElement('div');
            guessItem.className = 'guess-item ' + (data.is_correct ? 'correct' : 'incorrect');
            guessItem.textContent = data.guess + (data.is_correct ? ' ✓' : ' ✗');
            
            document.getElementById('guessHistory').appendChild(guessItem);
            document.getElementById('guessHistory2').appendChild(guessItem.cloneNode(true));
        }
        
        // 初始化WebSocket
        initWebSocket();
        
        // 回车键提交猜测
        document.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && role === 'guesser') {
                submitGuess();
            }
        });
        
        // 显示正确答案
        function showCorrectAnswer() {
            const correctAnswerDiv = document.getElementById('correctAnswer');
            correctAnswerDiv.textContent = `恭喜！正确答案是：${document.getElementById('currentWord').textContent}`;
            correctAnswerDiv.style.display = 'block';
        }
        
        // 隐藏正确答案
        function hideCorrectAnswer() {
            const correctAnswerDiv = document.getElementById('correctAnswer');
            correctAnswerDiv.style.display = 'none';
        }
    </script>
</body>
</html>
'''

with open("static/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

def get_local_ip():
    """获取本地IP地址"""
    try:
        # 创建一个UDP套接字，不实际连接，只是用来获取本地IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到一个外部地址，不会实际发送数据
        s.connect(("8.8.8.8", 80))
        # 获取本地IP地址
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"获取本地IP失败: {e}")
        return "127.0.0.1"

# 启动服务器
if __name__ == "__main__":
    print("正在启动服务器...")
    local_ip = get_local_ip()
    print(f"本地访问地址: http://localhost:8001/static/index.html")
    print(f"外部访问地址: http://{local_ip}:8001/static/index.html")
    print("其他设备可以通过上面的外部访问地址访问游戏")
    uvicorn.run(app, host="0.0.0.0", port=8001)

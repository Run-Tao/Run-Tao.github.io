import cv2
import mediapipe as mp
import numpy as np
import random
import time
import requests
import base64
import json
from PIL import ImageFont, ImageDraw, Image
import asyncio
import websockets

class GestureMultiplayerGame:
    def __init__(self):
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 手指位置平滑相关
        self.index_finger_history = []
        self.index_finger_history_max_len = 5  # 保存最近5帧的食指位置
        self.smoothing_factor = 0.7  # 平滑因子，0-1之间，越大越平滑
        
        # 绘制活动检测相关
        self.last_draw_activity_time = time.time()  # 上次绘制活动的时间
        self.draw_activity_threshold = 0.5  # 绘制活动阈值（秒），超过此时间未绘制则允许手势检测
        self.is_drawing_active = False  # 当前是否正在绘制活动中
        self.gesture_detection_enabled = True  # 当前是否允许手势检测
        self.last_gesture_time = time.time()  # 上次检测到手势的时间
        self.gesture_cooldown = 1.0  # 手势检测冷却时间（秒），避免频繁检测
        
        # 初始化画布
        self.canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 游戏状态
        self.drawing = False
        self.last_x, self.last_y = 0, 0
        
        # 颜色列表和颜色索引（只保留5种颜色）
        # 注意：OpenCV使用BGR颜色格式，所以顺序是（蓝，绿，红）
        self.colors = [
            (0, 0, 0),       # 黑色
            (0, 0, 255),     # 红色（BGR）
            (0, 255, 0),     # 绿色（BGR）
            (255, 0, 0),     # 蓝色（BGR）
            (0, 255, 255)    # 黄色（BGR）
        ]
        self.color_index = 0
        self.draw_color = self.colors[self.color_index]
        self.draw_thickness = 2
        
        # WebSocket客户端
        self.websocket = None
        self.ws_connected = False
        self.current_word = ""
        self.is_game_active = True
        self.guesses = []
        
        # 提示词功能
        self.hint = ""
        self.hint_input_active = False
        
        # 大模型API配置
        self.api_key = "sk-b0a9299097b04f7baef9f3254fc273e9"  # 替换为实际的API Key
        self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = "qwen-vl-plus"
        
        # AI猜测结果
        self.ai_guess = ""
        
        # 显示正确答案的状态
        self.show_correct_answer = False
    
    async def connect_to_server(self, server_url="wss://run-tao-github-io.onrender.com/ws"):  # 使用Render云服务器地址
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(server_url)
            self.ws_connected = True
            print(f"已连接到服务器: {server_url}")
            
            # 注册为画画的人
            await self.websocket.send(json.dumps({
                "type": "register",
                "role": "drawer"
            }))
            
            # 启动消息接收协程
            asyncio.create_task(self.receive_messages())
        except Exception as e:
            print(f"连接服务器失败: {e}")
            self.ws_connected = False
    
    async def receive_messages(self):
        """接收服务器消息"""
        try:
            while self.ws_connected:
                message = await self.websocket.recv()
                data = json.loads(message)
                await self.handle_message(data)
        except Exception as e:
            print(f"接收消息失败: {e}")
            self.ws_connected = False
            # 触发重连
            print("尝试重新连接服务器...")
            await self.connect_to_server()
    
    async def connect_to_server(self, server_url="wss://run-tao-github-io.onrender.com/ws"):  # 使用Render云服务器地址
        """连接到WebSocket服务器，支持自动重连"""
        retries = 0
        max_retries = 5
        retry_delay = 5  # 重连延迟时间（秒）
        
        while retries < max_retries and not self.ws_connected:
            try:
                self.websocket = await websockets.connect(server_url)
                self.ws_connected = True
                print(f"已连接到服务器: {server_url}")
                retries = 0  # 重置重试计数
                
                # 注册为画画的人
                await self.websocket.send(json.dumps({
                    "type": "register",
                    "role": "drawer"
                }))
                
                # 启动消息接收协程
                asyncio.create_task(self.receive_messages())
            except Exception as e:
                retries += 1
                self.ws_connected = False
                print(f"连接服务器失败 (尝试 {retries}/{max_retries}): {e}")
                if retries < max_retries:
                    print(f"{retry_delay}秒后尝试重新连接...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("达到最大重试次数，停止尝试")
                    break
    
    async def handle_message(self, data):
        """处理接收到的消息"""
        if data["type"] == "game_state":
            self.current_word = data["current_word"]
            self.is_game_active = data["is_game_active"]
            print(f"当前词: {self.current_word}")
        elif data["type"] == "guess_result":
            # 更新猜测记录
            self.guesses.append({
                "guess": data["guess"],
                "is_correct": data["is_correct"]
            })
            print(f"猜测: {data['guess']}, 结果: {'正确' if data['is_correct'] else '错误'}")
            
            # 如果猜测正确，显示正确答案
            if data["is_correct"]:
                self.show_correct_answer = True
        elif data["type"] == "game_reset":
            # 重置游戏
            self.current_word = data["current_word"]
            self.is_game_active = True
            self.guesses = []
            self.clear_canvas()
            self.show_correct_answer = False  # 重置显示正确答案状态
            print(f"游戏重置，新词: {self.current_word}")
    
    async def send_draw_update(self, x, y, drawing):
        """发送绘制更新到服务器，包含颜色信息"""
        if self.ws_connected:
            try:
                # 发送绘制更新，包含当前颜色
                await self.websocket.send(json.dumps({
                    "type": "draw",
                    "x": x,
                    "y": y,
                    "drawing": drawing,
                    "color": list(self.draw_color)  # 发送当前颜色，转换为列表格式
                }))
            except Exception as e:
                print(f"发送绘制更新失败: {e}")
                self.ws_connected = False
    
    async def send_clear_canvas(self):
        """发送清空画布命令到服务器"""
        if self.ws_connected:
            try:
                await self.websocket.send(json.dumps({
                    "type": "clear"
                }))
            except Exception as e:
                print(f"发送清空画布命令失败: {e}")
                self.ws_connected = False
    
    async def send_reset_game(self):
        """发送重置游戏命令到服务器"""
        if self.ws_connected:
            try:
                await self.websocket.send(json.dumps({
                    "type": "reset"
                }))
            except Exception as e:
                print(f"发送重置游戏命令失败: {e}")
                self.ws_connected = False
    
    def save_drawing(self):
        """保存画作到本地"""
        import os
        
        # 创建保存目录
        save_dir = "drawings"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 生成带有时间戳的文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f"{save_dir}/drawing_{timestamp}.jpg"
        
        # 保存画布内容
        cv2.imwrite(filename, self.canvas)
        
        return filename
    
    async def upload_drawing_to_server(self):
        """将当前画布上传到服务器，让猜词的人看到"""
        if self.ws_connected:
            try:
                # 将画布转换为base64编码
                _, buffer = cv2.imencode('.jpg', self.canvas)
                base64_str = base64.b64encode(buffer).decode('utf-8')
                
                # 发送画布更新消息
                await self.websocket.send(json.dumps({
                    "type": "canvas_update",
                    "canvas": base64_str
                }))
            except Exception as e:
                print(f"上传画布失败: {e}")
                self.ws_connected = False
    
    def draw_chinese_text(self, img, text, position, font_size=20, color=(255, 255, 255), bold=False):
        """在OpenCV图像上绘制中文文本"""
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试使用不同的中文字体
        font = None
        font_paths = []
        
        # 根据bold参数添加不同的字体路径
        if bold:
            # 优先尝试粗体字体
            font_paths.extend(["simhei.ttf", "msyhbd.ttc", "msyh.ttc"])
        else:
            # 普通字体
            font_paths.extend(["simhei.ttf", "msyh.ttc"])
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        # 如果所有字体都失败，使用默认字体
        if font is None:
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color)
        
        # 将PIL图像转换回OpenCV图像
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
    
    def clear_canvas(self):
        """清空画布"""
        self.canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self.ai_guess = ""  # 清空AI猜测
    
    def calculate_palm_center(self, hand_landmarks, w, h):
        """计算手掌中心位置"""
        # 选择多个手指根部关键点来计算手掌中心
        landmarks_to_use = [
            self.mp_hands.HandLandmark.WRIST,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]
        
        x_sum = 0
        y_sum = 0
        for landmark in landmarks_to_use:
            x_sum += hand_landmarks.landmark[landmark].x
            y_sum += hand_landmarks.landmark[landmark].y
        
        return int(x_sum / len(landmarks_to_use) * w), int(y_sum / len(landmarks_to_use) * h)
    
    def is_finger_extended(self, hand_landmarks, finger_tips, finger_mcps, w, h):
        """检测手指是否伸直"""
        extended = []
        for tip, mcp in zip(finger_tips, finger_mcps):
            # 获取指尖和指根的坐标
            tip_x, tip_y = int(hand_landmarks.landmark[tip].x * w), int(hand_landmarks.landmark[tip].y * h)
            mcp_x, mcp_y = int(hand_landmarks.landmark[mcp].x * w), int(hand_landmarks.landmark[mcp].y * h)
            
            # 对于拇指，判断其是否远离手掌中心
            if tip == self.mp_hands.HandLandmark.THUMB_TIP:
                # 获取手掌中心
                palm_x, palm_y = self.calculate_palm_center(hand_landmarks, w, h)
                # 拇指伸直的判断：指尖x坐标大于指根x坐标（右手）
                extended.append(tip_x > mcp_x + 20)
            else:
                # 其他手指伸直的判断：指尖y坐标小于指根y坐标（向上伸直）
                extended.append(tip_y < mcp_y - 20)
        
        return extended
    
    def recognize_gesture(self, hand_landmarks, w, h):
        """识别手势（数字1-5）"""
        # 定义手指的指尖和指根关键点
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_mcps = [
            self.mp_hands.HandLandmark.THUMB_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]
        
        # 检测哪些手指伸直
        extended = self.is_finger_extended(hand_landmarks, finger_tips, finger_mcps, w, h)
        
        # 统计伸直的手指数量
        extended_count = sum(extended)
        
        # 根据伸直的手指数量识别手势
        if extended_count == 1:
            if extended[1]:  # 只有食指伸直
                return 1
        elif extended_count == 2:
            if extended[1] and extended[2]:  # 食指和中指伸直
                return 2
        elif extended_count == 3:
            if extended[1] and extended[2] and extended[3]:  # 食指、中指和无名指伸直
                return 3
        elif extended_count == 4:
            if extended[1] and extended[2] and extended[3] and extended[4]:  # 食指、中指、无名指和小指伸直
                return 4
        elif extended_count == 5:
            # 所有手指都伸直
            return 5
        
        return None
    
    async def process_frame(self, frame):
        """处理每一帧"""
        # 翻转帧
        frame = cv2.flip(frame, 1)
        
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测手部
        results = self.hands.process(rgb_frame)
        
        # 绘制手部关键点
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                h, w, _ = frame.shape
                
                # 更新绘制活动状态
                current_time = time.time()
                if self.drawing:
                    # 正在绘制，更新活动时间
                    self.last_draw_activity_time = current_time
                    self.is_drawing_active = True
                else:
                    # 不在绘制，检查是否超过活动阈值
                    if current_time - self.last_draw_activity_time > self.draw_activity_threshold:
                        self.is_drawing_active = False
                    else:
                        self.is_drawing_active = True
                
                # 更新手势检测状态
                if current_time - self.last_gesture_time > self.gesture_cooldown:
                    self.gesture_detection_enabled = True
                else:
                    self.gesture_detection_enabled = False
                
                # 只在非绘制状态且允许手势检测时识别手势
                if not self.is_drawing_active and self.gesture_detection_enabled:
                    gesture = self.recognize_gesture(hand_landmarks, w, h)
                    
                    # 根据手势执行相应操作
                    if gesture is not None:
                        self.last_gesture_time = current_time  # 更新手势检测时间
                        self.gesture_detection_enabled = False  # 进入冷却状态
                        
                        if gesture == 1:
                            # 手势1：绘画模式（默认黑色）
                            self.draw_color = self.colors[0]  # 黑色
                            print("切换到绘画模式（黑色）")
                        elif gesture == 2:
                            # 手势2：切换到红色
                            self.draw_color = self.colors[1]  # 红色
                            print("切换颜色: 红色")
                        elif gesture == 3:
                            # 手势3：切换到绿色
                            self.draw_color = self.colors[2]  # 绿色
                            print("切换颜色: 绿色")
                        elif gesture == 4:
                            # 手势4：切换到蓝色
                            self.draw_color = self.colors[3]  # 蓝色
                            print("切换颜色: 蓝色")
                        elif gesture == 5:
                            # 手势5：切换到黄色
                            self.draw_color = self.colors[4]  # 黄色
                            print("切换颜色: 黄色")
                
                # 获取食指指尖坐标
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                raw_x, raw_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # 平滑处理食指位置
                self.index_finger_history.append((raw_x, raw_y))
                if len(self.index_finger_history) > self.index_finger_history_max_len:
                    self.index_finger_history.pop(0)
                
                # 计算平滑后的位置
                if len(self.index_finger_history) > 1:
                    # 加权平均平滑
                    x = int(self.index_finger_history[-1][0] * self.smoothing_factor + \
                            self.index_finger_history[-2][0] * (1 - self.smoothing_factor))
                    y = int(self.index_finger_history[-1][1] * self.smoothing_factor + \
                            self.index_finger_history[-2][1] * (1 - self.smoothing_factor))
                else:
                    x, y = raw_x, raw_y
                
                # 获取拇指指尖坐标
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                
                # 计算食指和拇指之间的距离
                distance = np.sqrt((x - thumb_x)**2 + (y - thumb_y)**2)
                
                # 根据距离判断是否开始绘制
                if distance < 40:  # 手指捏合，开始绘制
                    if not self.drawing:
                        self.drawing = True
                        self.last_x, self.last_y = x, y
                        self.last_draw_activity_time = time.time()  # 更新活动时间
                        await self.send_draw_update(x, y, True)
                    else:
                        # 绘制线条
                        # 降低移动距离阈值，让绘制更灵敏
                        move_distance = np.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
                        if move_distance > 0.5:  # 降低移动距离阈值，让绘制更灵敏
                            cv2.line(self.canvas, (self.last_x, self.last_y), (x, y), self.draw_color, self.draw_thickness)
                            self.last_x, self.last_y = x, y
                            self.last_draw_activity_time = time.time()  # 更新活动时间
                            await self.send_draw_update(x, y, True)
                        # 即使移动距离很小，也要更新活动时间
                        else:
                            self.last_draw_activity_time = time.time()
                else:  # 手指分开，停止绘制
                    if self.drawing:
                        self.drawing = False
                        # 清空食指历史记录
                        self.index_finger_history.clear()
                        await self.send_draw_update(x, y, False)
        # 显示当前颜色
        color_name = "黑色" if self.draw_color == self.colors[0] else "红色" if self.draw_color == self.colors[1] else "绿色" if self.draw_color == self.colors[2] else "蓝色" if self.draw_color == self.colors[3] else "黄色"
        frame = self.draw_chinese_text(frame, f"当前颜色: {color_name}", (10, 430), font_size=20, color=(0, 255, 255))
        
        # 显示游戏信息
        frame = self.draw_chinese_text(frame, f"目标词: {self.current_word}", (10, 30), font_size=20, color=(255, 0, 0))
        frame = self.draw_chinese_text(frame, "捏合手指开始绘制", (10, 70), font_size=14, color=(0, 255, 0))
        frame = self.draw_chinese_text(frame, "按 'c' 清空画布", (10, 110), font_size=14, color=(0, 255, 0))
        frame = self.draw_chinese_text(frame, "按 'h' 输入提示词", (10, 150), font_size=14, color=(0, 255, 0))
        frame = self.draw_chinese_text(frame, "按 'g' 让AI猜测", (10, 190), font_size=14, color=(0, 255, 0))
        frame = self.draw_chinese_text(frame, "按 'f' 保存并上传", (10, 230), font_size=14, color=(0, 255, 0))
        frame = self.draw_chinese_text(frame, "按 'r' 重新开始", (10, 270), font_size=14, color=(0, 255, 0))
        frame = self.draw_chinese_text(frame, "按 'q' 退出游戏", (10, 310), font_size=14, color=(0, 255, 0))
        
        # 显示AI猜测结果
        if self.ai_guess:
            frame = self.draw_chinese_text(frame, f"AI猜测: {self.ai_guess}", (10, 350), font_size=20, color=(0, 0, 255))
        
        # 显示提示词
        if self.hint:
            frame = self.draw_chinese_text(frame, f"提示词: {self.hint}", (10, 390), font_size=16, color=(255, 0, 0))
        
        # 显示猜测记录
        if self.guesses:
            start_y = 420 if self.hint else 390
            frame = self.draw_chinese_text(frame, "猜测记录:", (10, start_y), font_size=16, color=(255, 165, 0))
            for i, guess in enumerate(self.guesses[-3:]):  # 只显示最近3个猜测
                result = "正确" if guess["is_correct"] else "错误"
                frame = self.draw_chinese_text(frame, f"{i+1}. {guess['guess']} - {result}", (10, start_y + 30 + i*30), font_size=14, color=(255, 165, 0))
        
        # 显示正确答案（当猜测正确时）
        if self.show_correct_answer:
            # 计算显示位置，确保不与其他信息重叠
            correct_answer_y = 420 + len(self.guesses[-3:]) * 30 if (self.hint and self.guesses) else 390 + len(self.guesses[-3:]) * 30
            frame = self.draw_chinese_text(frame, f"正确答案: {self.current_word}", (10, correct_answer_y), font_size=24, color=(0, 255, 0), bold=True)
        
        # 合并画布和摄像头画面
        combined = np.hstack((frame, self.canvas))
        
        return combined
    
    async def run(self):
        """运行游戏"""
        # 连接到服务器
        await self.connect_to_server()
        
        # 尝试不同的摄像头索引，解决无法打开摄像头的问题
        cap = None
        for i in range(3):  # 尝试0、1、2三个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"成功打开摄像头，索引: {i}")
                break
            else:
                cap.release()
                cap = None
                print(f"尝试打开摄像头 {i} 失败")
        
        if not cap:
            print("无法打开任何摄像头，请检查摄像头连接和权限")
            return
        
        # 设置较低的摄像头分辨率，提高处理速度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 获取实际设置的分辨率
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头分辨率: {actual_width}x{actual_height}")
        
        print("游戏开始！捏合手指开始绘制，按 'c' 清空画布，按 'h' 输入提示词，按 'g' 让AI猜测，按 'r' 重新开始，按 'q' 退出游戏")
        
        frame_count = 0
        error_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                error_count += 1
                print(f"第 {frame_count+1} 帧：无法获取视频帧 (连续错误: {error_count})")
                if error_count > 10:  # 连续10帧错误则退出
                    print("连续获取视频帧失败，退出游戏")
                    break
                continue  # 继续尝试获取下一帧
            
            error_count = 0  # 重置错误计数
            frame_count += 1
            
            # 处理帧
            combined = await self.process_frame(frame)
            
            # 显示画面
            cv2.imshow('手势绘画游戏', combined)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # 清空画布
                self.clear_canvas()
                await self.send_clear_canvas()
                print("画布已清空")
            elif key == ord('h'):  # 输入提示词
                hint = input("请输入提示词（按Enter确认）: ")
                self.hint = hint.strip()
                print(f"提示词已设置: {self.hint}")
                # 自动让AI猜测
                print("正在根据提示词猜测...")
                self.ai_guess = self.guess_drawing()
                print(f"AI猜测: {self.ai_guess}")
            elif key == ord('g'):  # 让AI猜测
                print("正在猜测...")
                self.ai_guess = self.guess_drawing()
                print(f"AI猜测: {self.ai_guess}")
            elif key == ord('r'):  # 重新开始
                await self.send_reset_game()
            elif key == ord('f'):  # 保存图片并上传到前端
                filename = self.save_drawing()
                print(f"图片已保存到: {filename}")
                await self.upload_drawing_to_server()
                print("图片已上传到前端")
            elif key == ord('q'):  # 退出游戏
                print("游戏结束！")
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 关闭WebSocket连接
        if self.ws_connected:
            await self.websocket.close()
            self.ws_connected = False
    
    def canvas_to_base64(self):
        """将画布转换为base64编码"""
        # 将画布转换为JPEG格式
        _, buffer = cv2.imencode('.jpg', self.canvas)
        # 转换为base64编码
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return base64_str
    
    def guess_drawing(self):
        """调用大模型猜测绘制内容"""
        try:
            # 将画布转换为base64
            base64_image = self.canvas_to_base64()
            
            # 构建API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 构建用户提示词，包含可选的提示词
            user_prompt = "请根据这张图片中的手绘内容，猜测画的是什么物体。"
            if self.hint:
                user_prompt += f" 提示词: {self.hint}"
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的图像识别助手，请根据图片中的手绘内容，猜测画的是什么物体。请只用一个词或短语回答，不要有任何解释。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 20
            }
            
            # 发送请求
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            guess = result["choices"][0]["message"]["content"].strip()
            
            return guess
        except Exception as e:
            print(f"猜测失败: {e}")
            return "猜测失败，请重试"

if __name__ == "__main__":
    import asyncio
    game = GestureMultiplayerGame()
    asyncio.run(game.run())

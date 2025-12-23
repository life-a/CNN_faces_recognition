import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import dlib
import numpy as np
import tensorflow.compat.v1 as tf
import os
import time
from PIL import Image, ImageTk
import glob
from collections import deque, Counter
import random
from datetime import datetime


def layer_net(input_image, num_class, dropout_rate, dropout_rate_2):
    """完全按照原代码定义"""
    tf.disable_eager_execution()

    """第一、二层，输入图片64*64*3，输出图片32*32*32"""
    w1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), name='w1')  # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = tf.Variable(tf.random.normal([32]), name='b1')
    layer_conv1 = tf.nn.relu(
        tf.nn.conv2d(input_image, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)  # 64*64*32，卷积提取特征，增加通道数
    layer_pool1 = tf.nn.max_pool2d(layer_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')  # 32*32*32，池化降维，减小复杂度
    drop1 = tf.nn.dropout(layer_pool1, rate=1 - dropout_rate)  # 按一定概率随机丢弃一些神经元，以获得更高的训练速度以及防止过拟合

    """第三、四层，输入图片32*32*32，输出图片16*16*64"""
    w2 = tf.Variable(tf.random.normal([3, 3, 32, 64]), name='w2')  # 卷积核大小(3,3)， 输入通道(32)， 输出通道(64)
    b2 = tf.Variable(tf.random.normal([64]), name='b2')
    layer_conv2 = tf.nn.relu(tf.nn.conv2d(drop1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)  # 32*32*64
    layer_pool2 = tf.nn.max_pool2d(layer_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 16*16*64
    drop2 = tf.nn.dropout(layer_pool2, rate=1 - dropout_rate)

    """第五、六层，输入图片16*16*64，输出图片8*8*64"""
    w3 = tf.Variable(tf.random.normal([3, 3, 64, 64]), name='w3')  # 卷积核大小(3,3)， 输入通道(64)， 输出通道(64)
    b3 = tf.Variable(tf.random.normal([64]), name='b3')
    layer_conv3 = tf.nn.relu(tf.nn.conv2d(drop2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)  # 16*16*64
    layer_pool3 = tf.nn.max_pool2d(layer_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')  # 8*8*64=4096
    drop3 = tf.nn.dropout(layer_pool3, rate=1 - dropout_rate)

    """第七层，全连接层，将图片的卷积输出压扁成一个一维向量，输入图片8*8*64，reshape到1*4096，输出1*512"""
    w4 = tf.Variable(tf.random.normal([8 * 8 * 64, 512]), name='w4')  # 输入通道(4096)， 输出通道(512)
    b4 = tf.Variable(tf.random.normal([512]), name='b4')
    layer_fully_connected = tf.reshape(drop3, [-1, 8 * 8 * 64])  # -1表示行随着列的需求改变，1*4096
    relu = tf.nn.relu(tf.matmul(layer_fully_connected, w4) + b4)  # [1,4096]*[4096,512]=[1,512]
    drop4 = tf.nn.dropout(relu, rate=1 - dropout_rate_2)

    """第八层，输出层，输入1*512，输出1*2，再add"""
    w5 = tf.Variable(tf.random.normal([512, num_class]), name='w5')  # 输入通道(512)， 输出通道(2)
    b5 = tf.Variable(tf.random.normal([num_class]), name='b5')
    outdata = tf.add(tf.matmul(drop4, w5), b5)  # (1,512)*(512,2)=(1,2) ,跟input_label [0,1]、[1,0]比较给出损失 ，先乘再加
    return outdata


class FaceDataCollector:
    """人脸数据采集器（核心功能版）"""

    def __init__(self, size=64):
        self.size = size
        self.detector = dlib.get_frontal_face_detector()

    def apply_augmentations(self, img):
        """应用三种数据增强"""
        augmentations = []

        # 1. 水平翻转
        flipped = cv2.flip(img, 1)
        augmentations.append(flipped)

        # 2. 亮度调整（随机变亮或变暗）
        alpha = random.uniform(0.7, 1.3)  # 对比度
        beta = random.randint(-30, 30)    # 亮度
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        augmentations.append(bright)

        # 3. 对比度调整
        contrast = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=0)
        augmentations.append(contrast)

        return augmentations

    def capture_data(self, person_name, target_count=50, cap=None):
        """
        采集指定人员的人脸数据
        :param person_name: 人员名称
        :param target_count: 目标采集数量
        :param cap: 摄像头对象
        """
        # 创建保存目录：faces_ok/人员名称/
        save_dir = os.path.join('./faces_ok', person_name)
        os.makedirs(save_dir, exist_ok=True)

        # 检查已有图片数量
        existing_files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))]
        saved_count = len(existing_files)

        # 打开摄像头（如果未提供）
        if cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print(f"❌ 无法打开摄像头")
                return False

        frame_skip = 3  # 每3帧采集一次，避免过于相似
        frame_counter = 0

        while saved_count < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            # 人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)

            # 处理检测到的人脸
            for face in faces:
                x1 = max(face.top(), 0)
                y1 = min(face.bottom(), frame.shape[0])
                x2 = max(face.left(), 0)
                y2 = min(face.right(), frame.shape[1])

                face_img = frame[x1:y1, x2:y2]
                if face_img.size > 0 and face_img.shape[0] > 20 and face_img.shape[1] > 20:
                    # 调整到标准尺寸
                    face_resized = cv2.resize(face_img, (self.size, self.size))

                    # 生成时间戳作为文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    base_filename = f"{person_name}_{timestamp}"

                    # 保存原始图片
                    original_path = os.path.join(save_dir, f"{base_filename}_orig.jpg")
                    cv2.imwrite(original_path, face_resized)
                    saved_count += 1

                    # 自动生成3个增强版本
                    if saved_count < target_count:
                        augmentations = self.apply_augmentations(face_resized)
                        for i, aug_img in enumerate(augmentations):
                            if saved_count >= target_count:
                                break

                            aug_path = os.path.join(save_dir, f"{base_filename}_aug{i+1}.jpg")
                            cv2.imwrite(aug_path, aug_img)
                            saved_count += 1

                    # 更新进度
                    progress = min(saved_count / target_count, 1.0)
                    print(f"✅ 已保存 {saved_count}/{target_count} (进度: {progress*100:.1f}%)")
                    break  # 只处理第一张脸

            if frame_counter % 10 == 0:  # 每10帧检查一次
                break  # 在GUI中，我们只采集一帧然后返回

        return True


class ImprovedFaceRecognizer:
    """改进的人脸识别器，解决总是识别为同一个人的问题"""

    def __init__(self, model_path='./model_multi_class/'):
        self.model_path = model_path
        self.sess = None
        self.outdata = None
        self.input_image = None
        self.dropout_rate = None
        self.dropout_rate_2 = None
        self.class_names = []
        self.num_classes = 0

        # 时间平滑参数
        self.prediction_history = deque(maxlen=15)  # 保存最近15次预测
        self.confidence_history = deque(maxlen=15)   # 保存最近15次置信度

        # 动态阈值参数
        self.base_threshold = 0.65  # 基础置信度阈值
        self.class_thresholds = {}   # 每个类别的动态阈值

    def load_model(self):
        """加载训练好的平衡模型"""
        print(f"正在从 {self.model_path} 加载模型...")

        # 检查模型目录是否存在
        if not os.path.exists(self.model_path):
            print(f"错误: 模型目录 {self.model_path} 不存在")
            return False

        # 尝试查找最新的模型checkpoint
        checkpoint_path = self.find_latest_model()
        if checkpoint_path is None:
            print("未找到模型checkpoint")
            return False

        # 读取类别名称
        class_names_file = './model_multi_class/class_names.txt'
        if os.path.exists(class_names_file):
            with open(class_names_file, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # 尝试从faces_ok目录推断
            faces_ok_dir = './faces_ok'
            if os.path.exists(faces_ok_dir):
                class_names = []
                for item in os.listdir(faces_ok_dir):
                    item_path = os.path.join(faces_ok_dir, item)
                    if os.path.isdir(item_path):
                        class_names.append(item)
                # 添加陌生人类别
                class_names.append("陌生人")
                self.class_names = class_names
            else:
                print("faces_ok目录不存在，使用默认类别")
                self.class_names = ["我的人脸", "其他人脸"]

        self.num_classes = len(self.class_names)
        print(f"加载了 {self.num_classes} 个类别: {self.class_names}")

        # 为每个类别初始化动态阈值
        for i, name in enumerate(self.class_names):
            if name == "陌生人":
                self.class_thresholds[i] = 0.55  # 陌生人阈值较低
            else:
                self.class_thresholds[i] = 0.65  # 已知人员阈值较高

        # 定义占位符
        size = 64
        self.input_image = tf.placeholder(tf.float32, [None, size, size, 3], name='input_image')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
        self.dropout_rate_2 = tf.placeholder(tf.float32, name='dropout_rate_2')

        # 构建网络
        self.outdata = layer_net(self.input_image, self.num_classes,
                                 self.dropout_rate, self.dropout_rate_2)

        # 创建会话
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        saver = tf.train.Saver()

        try:
            saver.restore(self.sess, checkpoint_path)
            print("模型加载成功")

            # 测试模型是否正常工作
            test_input = np.random.randn(1, 64, 64, 3) * 0.1
            probs = self.sess.run(tf.nn.softmax(self.outdata),
                                 feed_dict={
                                     self.input_image: test_input,
                                     self.dropout_rate: 1.0,
                                     self.dropout_rate_2: 1.0
                                 })
            print(f"模型测试通过，输出shape: {probs.shape}")
            return True

        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_latest_model(self):
        """查找最新的best_model文件"""
        model_dir = './model_multi_class/'
        if not os.path.exists(model_dir):
            print(f"模型目录 {model_dir} 不存在")
            return None

        # 首先尝试从checkpoint文件获取模型路径
        try:
            checkpoint_file = os.path.join(model_dir, 'checkpoint')
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'model_checkpoint_path' in line:
                            model_name = line.split('"')[1]  # 提取模型名称
                            model_path = os.path.join(model_dir, model_name)
                            print(f"从checkpoint文件找到模型路径: {model_path}")
                            return model_path
        except Exception as e:
            print(f"读取checkpoint文件失败: {e}")

        # 如果checkpoint文件不存在，手动查找best_model文件
        pattern = os.path.join(model_dir, 'best_model-*')
        files = glob.glob(pattern)

        if not files:
            print("未找到任何best_model文件")
            return None

        # 按步数排序，找到步数最大的
        model_steps = []
        for f in files:
            try:
                # 提取步数（文件名格式: best_model-步数.meta/data/index）
                base_name = os.path.basename(f)
                if '-' in base_name:
                    step_str = base_name.split('-')[1].split('.')[0]
                    if step_str.isdigit():
                        step = int(step_str)
                        model_steps.append((step, f))
            except:
                continue

        if not model_steps:
            print("无法从文件名中提取步数")
            return None

        # 返回步数最大的模型
        max_step, max_file = max(model_steps, key=lambda x: x[0])
        # 获取基本路径（不含扩展名）
        base_path = max_file.split('.')[0]
        print(f"找到最新模型: {base_path} (步数: {max_step})")
        return base_path

    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        if face_img is None or face_img.size == 0:
            return None

        # 调整大小到64x64
        face_resized = cv2.resize(face_img, (64, 64))

        # 直方图均衡化（增强对比度）
        img_yuv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        face_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # 轻微高斯模糊去噪
        face_blur = cv2.GaussianBlur(face_eq, (3, 3), 0)

        return face_blur

    def recognize_single_frame(self, face_img):
        """识别单帧中的人脸"""
        if face_img is None:
            return None, 0.0, None

        # 预处理
        processed_face = self.preprocess_face(face_img)
        if processed_face is None:
            return None, 0.0, None

        # 归一化
        face_normalized = processed_face.astype(np.float32) / 255.0

        try:
            # 运行推理
            logits = self.sess.run(self.outdata,
                                  feed_dict={
                                      self.input_image: [face_normalized],
                                      self.dropout_rate: 1.0,  # 推理时dropout率为1.0（不使用dropout）
                                      self.dropout_rate_2: 1.0  # 推理时dropout率为1.0（不使用dropout）
                                  })

            # 计算softmax概率
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            probs = probs[0]  # 取第一个样本

            # 获取原始预测结果
            raw_predicted = np.argmax(probs)
            raw_confidence = np.max(probs)

            return raw_predicted, raw_confidence, probs

        except Exception as e:
            print(f"推理错误: {e}")
            return None, 0.0, None

    def recognize_with_smoothing(self, face_img):
        """使用时间平滑的识别人脸"""
        # 获取当前帧的识别结果
        raw_predicted, raw_confidence, probs = self.recognize_single_frame(face_img)

        if raw_predicted is None:
            return None, 0.0, None

        # 保存到历史
        self.prediction_history.append(raw_predicted)
        self.confidence_history.append(raw_confidence)

        # 应用时间平滑：使用历史投票
        final_predicted = raw_predicted
        final_confidence = raw_confidence

        if len(self.prediction_history) >= 5:
            # 计算最近5次预测的众数
            recent_predictions = list(self.prediction_history)[-5:]
            pred_counter = Counter(recent_predictions)
            most_common_pred, most_common_count = pred_counter.most_common(1)[0]

            # 如果众数出现次数超过3次，且与当前预测不同
            if most_common_count >= 3 and most_common_pred != raw_predicted:
                final_predicted = most_common_pred
                # 计算该类别在历史中的平均置信度
                confidences = [conf for pred, conf in
                              zip(list(self.prediction_history), list(self.confidence_history))
                              if pred == most_common_pred]
                final_confidence = np.mean(confidences) if confidences else raw_confidence

        # 应用动态阈值
        class_threshold = self.class_thresholds.get(final_predicted, self.base_threshold)

        # 如果置信度低于类别阈值，则认为是陌生人
        if final_confidence < class_threshold:
            # 找到陌生人对应的类别索引
            stranger_idx = None
            for i, name in enumerate(self.class_names):
                if name == "陌生人":
                    stranger_idx = i
                    break

            if stranger_idx is not None:
                final_predicted = stranger_idx
                final_confidence = probs[stranger_idx] if probs is not None else 0.0

        return final_predicted, final_confidence, probs

    def get_class_color(self, class_idx, confidence):
        """根据类别和置信度获取显示颜色"""
        if class_idx >= len(self.class_names):
            return (128, 128, 128)  # 灰色 - 未知类别

        class_name = self.class_names[class_idx]

        if class_name == "陌生人":
            if confidence > 0.7:
                return (0, 0, 255)  # 红色 - 高置信度陌生人
            else:
                return (0, 165, 255)  # 橙色 - 低置信度陌生人
        else:
            if confidence > 0.75:
                return (0, 255, 0)  # 绿色 - 高置信度已知人员
            elif confidence > 0.6:
                return (255, 255, 0)  # 黄色 - 中等置信度
            else:
                return (255, 165, 0)  # 橙色 - 低置信度

    def close(self):
        """关闭资源"""
        if self.sess:
            self.sess.close()


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统 - 改进布局版")
        self.root.geometry("1200x800")  # 增大窗口

        # 初始化变量
        self.cap = None
        self.is_running = False
        self.is_collecting = False
        self.collection_count = 0
        self.max_collection = 50
        self.current_user = "default_user"

        # 初始化人脸识别器
        self.face_recognizer = ImprovedFaceRecognizer()
        
        # 初始化人脸采集器
        self.face_collector = FaceDataCollector()
        
        # 加载人脸检测器（与faces_my.py一致）
        try:
            self.detector = dlib.get_frontal_face_detector()
        except Exception as e:
            messagebox.showerror("错误", f"无法加载dlib人脸检测器: {e}\n请确保已安装dlib库")
            raise

        # 检查项目目录结构
        self.check_project_structure()

        # 尝试加载模型
        self.model_loaded = self.face_recognizer.load_model()

        # 创建界面
        self.create_widgets()

        # 立即启动摄像头
        self.start_default_camera()

    def check_project_structure(self):
        """检查项目目录结构"""
        print("检查项目目录结构...")
        dirs_to_check = ['faces_ok', 'faces_no', 'model_multi_class']
        for dir_name in dirs_to_check:
            exists = os.path.exists(dir_name)
            print(f"目录 '{dir_name}': {'存在' if exists else '不存在'}")

    def start_default_camera(self):
        """启动默认摄像头"""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.is_running = True
        self.is_collecting = False  # 默认为识别模式
        self.recognize_btn.config(text="停止识别")
        self.collect_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="状态: 摄像头已启动（识别模式）")

        # 更新信息栏
        self.update_info("摄像头已启动...")
        self.update_video()

    def create_widgets(self):
        # 主框架 - 分左右两列
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：视频显示区域 (占60%宽度)
        left_frame = ttk.Frame(main_frame, width=720)  # 1200 * 0.6 = 720
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left_frame.pack_propagate(False)  # 固定宽度

        # 视频显示区域
        video_frame = ttk.LabelFrame(left_frame, text="实时画面", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame, background="black", text="摄像头已启动")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 右侧：控制面板和信息区域 (占40%宽度)
        right_frame = ttk.Frame(main_frame, width=480)  # 1200 * 0.4 = 480
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)  # 固定宽度

        # 项目信息和模型状态
        info_frame = ttk.LabelFrame(right_frame, text="项目信息", padding=5)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        # 检查并显示项目状态
        status_text = "项目状态:\n"
        status_text += f"- faces_ok: {'✓' if os.path.exists('./faces_ok') else '✗'}\n"
        status_text += f"- faces_no: {'✓' if os.path.exists('./faces_no') else '✗'}\n"
        status_text += f"- model_multi_class: {'✓' if os.path.exists('./model_multi_class') else '✗'}\n"

        ttk.Label(info_frame, text=status_text, justify=tk.LEFT, font=("Arial", 9)).pack(anchor=tk.W)

        # 模型状态
        model_status_frame = ttk.LabelFrame(right_frame, text="模型状态", padding=5)
        model_status_frame.pack(fill=tk.X, pady=(0, 10))

        if self.model_loaded:
            status_text = f"✓ 模型加载成功!\n- 类别数: {self.face_recognizer.num_classes}\n- 类别名称: {self.face_recognizer.class_names}"
            ttk.Label(model_status_frame, text=status_text, foreground="green", font=("Arial", 9)).pack(anchor=tk.W)
        else:
            status_text = "✗ 未检测到训练好的模型\n💡 请先运行 faces_train_multi_person.py 训练模型"
            ttk.Label(model_status_frame, text=status_text, foreground="red", font=("Arial", 9)).pack(anchor=tk.W)

        # 控制面板
        control_frame = ttk.LabelFrame(right_frame, text="控制面板", padding=5)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 用户名输入
        username_frame = ttk.Frame(control_frame)
        username_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(username_frame, text="用户名:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.user_entry = ttk.Entry(username_frame, width=15, font=("Arial", 9))
        self.user_entry.insert(0, self.current_user)
        self.user_entry.pack(side=tk.RIGHT)

        # 按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        self.collect_btn = ttk.Button(btn_frame, text="人脸采集", command=self.toggle_collection, width=10)
        self.collect_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.recognize_btn = ttk.Button(btn_frame, text="人脸识别", command=self.toggle_recognition, width=10)
        self.recognize_btn.pack(side=tk.LEFT, padx=(0, 5))
        if not self.model_loaded:
            self.recognize_btn.config(state=tk.DISABLED)

        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop_camera, state=tk.NORMAL, width=10)
        self.stop_btn.pack(side=tk.RIGHT)

        self.status_label = ttk.Label(control_frame, text="状态: 摄像头已启动", font=("Arial", 9))
        self.status_label.pack(pady=(5, 0))

        # 进度条
        self.progress = ttk.Progressbar(control_frame, mode='determinate', length=200)
        self.progress['maximum'] = self.max_collection
        self.progress.pack(pady=(5, 0), fill=tk.X)
        self.progress.pack_forget()  # 默认隐藏

        # 日志信息区域
        log_frame = ttk.LabelFrame(right_frame, text="日志信息", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.info_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Arial", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def toggle_collection(self):
        if not self.is_running:
            self.current_user = self.user_entry.get().strip()
            if not self.current_user:
                self.current_user = "default_user"
            self.start_collection()
        else:
            self.stop_camera()

    def toggle_recognition(self):
        if not self.is_running:
            self.start_recognition()
        else:
            self.stop_camera()

    def start_collection(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.is_running = True
        self.is_collecting = True
        self.collection_count = 0
        self.collect_btn.config(text="停止采集")
        self.recognize_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="状态: 正在采集人脸")
        self.progress.pack(pady=(5, 0), fill=tk.X)  # 显示进度条

        # 更新信息栏
        self.update_info(f"开始采集用户 '{self.current_user}' 的人脸数据...")
        self.update_video()

    def start_recognition(self):
        if not self.model_loaded:
            messagebox.showwarning("警告", "模型未加载，无法进行人脸识别")
            return

        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.is_running = True
        self.is_collecting = False
        self.recognize_btn.config(text="停止识别")
        self.collect_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="状态: 正在识别人脸")

        # 更新信息栏
        self.update_info("开始人脸识别...")
        self.update_video()

    def stop_camera(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.collect_btn.config(text="人脸采集", state=tk.NORMAL)
        self.recognize_btn.config(text="人脸识别", state=tk.NORMAL if self.model_loaded else tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="状态: 已停止")
        self.progress.pack_forget()  # 隐藏进度条

        # 清空进度条
        self.progress['value'] = 0

        # 清空视频标签
        self.video_label.configure(image='')
        self.video_label.configure(text="摄像头已停止")

        # 更新信息栏
        self.update_info("摄像头已停止")

    def update_video(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_video)
            return

        # 检测人脸（与faces_my.py一致）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)

        for i, d in enumerate(faces):
            x1 = max(d.top(), 0)
            y1 = min(d.bottom(), frame.shape[0])
            x2 = max(d.left(), 0)
            y2 = min(d.right(), frame.shape[1])

            # 绘制人脸框
            if self.is_collecting:
                cv2.rectangle(frame, (x2, x1), (y2, y1), (0, 255, 0), 2)  # 绿色框
            else:
                cv2.rectangle(frame, (x2, x1), (y2, y1), (255, 0, 0), 2)  # 蓝色框

            if self.is_collecting:
                # 采集模式：检测到人脸时保存照片（与faces_my.py一致）
                face_img = frame[x1:y1, x2:y2]
                if face_img.size > 0 and face_img.shape[0] > 20 and face_img.shape[1] > 20:  # 确保人脸足够大
                    face_resized = cv2.resize(face_img, (64, 64))

                    # 使用采集器保存数据
                    success = self.face_collector.capture_data(self.current_user, self.max_collection, self.cap)
                    if success:
                        self.collection_count += 1
                        self.progress['value'] = self.collection_count

                        if self.collection_count >= self.max_collection:
                            self.stop_camera()
                            messagebox.showinfo("提示",
                                                f"人脸采集完成，共采集{self.max_collection}张照片\n保存至: ./faces_ok/{self.current_user}/")
                            break
            elif self.model_loaded:
                # 识别模式：识别人脸（与face_recognition_multi_person.py一致）
                face_img = frame[x1:y1, x2:y2]
                if face_img.size > 0 and face_img.shape[0] > 20 and face_img.shape[1] > 20:  # 确保人脸足够大
                    # 识别人脸（使用face_recognition_multi_person.py的改进识别）
                    predicted_class_idx, confidence, all_probs = self.face_recognizer.recognize_with_smoothing(face_img)

                    # 获取类别名称
                    if predicted_class_idx < len(self.face_recognizer.class_names):
                        person_name = self.face_recognizer.class_names[predicted_class_idx]
                    else:
                        person_name = "未知"

                    # 根据置信度设置标签
                    if confidence > 0.55:
                        if person_name == "陌生人" or "其他" in person_name:
                            color = (0, 0, 255)  # 红色 - 陌生人
                            label = f"陌生人 ({confidence:.2f})"
                        else:
                            color = (0, 255, 0)  # 绿色 - 已知人员
                            label = f"{person_name} ({confidence:.2f})"
                    else:
                        color = (128, 128, 128)  # 灰色 - 置信度低
                        label = "低置信度"

                    # 在画面上显示识别结果
                    cv2.putText(frame, label, (x2, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # 更新信息显示
                    info_msg = f"识别结果: {person_name}, 置信度: {confidence:.3f}"
                    self.update_info(info_msg)

        # 转换为PIL图像并显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_pil = img_pil.resize((700, 500), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.video_label.img_tk = img_tk  # 保持引用
        self.video_label.configure(image=img_tk)

        # 每10毫秒更新一次
        self.root.after(10, self.update_video)

    def update_info(self, message):
        """更新信息显示区域"""
        current_time = time.strftime("%H:%M:%S", time.localtime())
        self.info_text.insert(tk.END, f"[{current_time}] {message}\n")
        self.info_text.see(tk.END)

    def close_app(self):
        self.stop_camera()
        if hasattr(self, 'face_recognizer'):
            self.face_recognizer.close()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()


if __name__ == "__main__":
    main()




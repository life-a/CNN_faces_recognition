"""
face_recognition_tool.py - 人脸识别工具类
重构人脸识别功能，使其可复用
"""

import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import glob
from collections import deque, Counter
import dlib
from PIL import Image, ImageDraw, ImageFont
import face_net_tool
tf.disable_v2_behavior()  # 禁用TensorFlow 2.x行为


# def layer_net(input_image, num_class, dropout_rate, dropout_rate_2):
#     """神经网络层定义"""
#     tf.disable_eager_execution()
#
#     """第一、二层，输入图片64*64*3，输出图片32*32*32"""
#     w1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), name='w1')
#     b1 = tf.Variable(tf.random.normal([32]), name='b1')
#     layer_conv1 = tf.nn.relu(
#         tf.nn.conv2d(input_image, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
#     layer_pool1 = tf.nn.max_pool2d(layer_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
#                                    padding='SAME')
#     drop1 = tf.nn.dropout(layer_pool1, rate=1 - dropout_rate)
#
#     """第三、四层，输入图片32*32*32，输出图片16*16*64"""
#     w2 = tf.Variable(tf.random.normal([3, 3, 32, 64]), name='w2')
#     b2 = tf.Variable(tf.random.normal([64]), name='b2')
#     layer_conv2 = tf.nn.relu(tf.nn.conv2d(drop1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
#     layer_pool2 = tf.nn.max_pool2d(layer_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     drop2 = tf.nn.dropout(layer_pool2, rate=1 - dropout_rate)
#
#     """第五、六层，输入图片16*16*64，输出图片8*8*64"""
#     w3 = tf.Variable(tf.random.normal([3, 3, 64, 64]), name='w3')
#     b3 = tf.Variable(tf.random.normal([64]), name='b3')
#     layer_conv3 = tf.nn.relu(tf.nn.conv2d(drop2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
#     layer_pool3 = tf.nn.max_pool2d(layer_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
#                                    padding='SAME')
#     drop3 = tf.nn.dropout(layer_pool3, rate=1 - dropout_rate)
#
#     """第七层，全连接层"""
#     w4 = tf.Variable(tf.random.normal([8 * 8 * 64, 512]), name='w4')
#     b4 = tf.Variable(tf.random.normal([512]), name='b4')
#     layer_fully_connected = tf.reshape(drop3, [-1, 8 * 8 * 64])
#     relu = tf.nn.relu(tf.matmul(layer_fully_connected, w4) + b4)
#     drop4 = tf.nn.dropout(relu, rate=1 - dropout_rate_2)
#
#     """第八层，输出层"""
#     w5 = tf.Variable(tf.random.normal([512, num_class]), name='w5')
#     b5 = tf.Variable(tf.random.normal([num_class]), name='b5')
#     outdata = tf.add(tf.matmul(drop4, w5), b5)
#     return outdata


class FaceRecognitionTool:
    """人脸识别工具类"""

    def __init__(self, model_path='./model_multi_class/'):
        """
        初始化人脸识别器

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.sess = None
        self.outdata = None
        self.input_image = None
        self.dropout_rate = None
        self.dropout_rate_2 = None
        self.class_names = []
        self.num_classes = 0

        # 时间平滑参数
        self.prediction_history = deque(maxlen=15)
        self.confidence_history = deque(maxlen=15)

        # 动态阈值参数
        self.base_threshold = 0.65
        self.class_thresholds = {}

        # 回调函数
        self.on_recognition_result = None
        self.on_info_update = None

    def load_model(self):
        """加载训练好的模型"""
        print(f"正在从 {self.model_path} 加载模型...")

        # 检查模型目录是否存在
        if not os.path.exists(self.model_path):
            print(f"错误: 模型目录 {self.model_path} 不存在")
            return False

        # 读取类别名称
        class_names_file = os.path.join(self.model_path, 'class_names.txt')
        if os.path.exists(class_names_file):
            with open(class_names_file, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # 尝试从faces_user目录推断
            faces_user_dir = './faces_user'
            if os.path.exists(faces_user_dir):
                class_names = []
                for item in os.listdir(faces_user_dir):
                    item_path = os.path.join(faces_user_dir, item)
                    if os.path.isdir(item_path):
                        class_names.append(item)
                # 添加陌生人类别
                class_names.append("陌生人")
                self.class_names = class_names
            else:
                print("faces_user目录不存在，使用默认类别")
                self.class_names = ["我的人脸", "其他人脸"]

        self.num_classes = len(self.class_names)
        print(f"加载了 {self.num_classes} 个类别: {self.class_names}")

        # 为每个类别初始化动态阈值
        for i, name in enumerate(self.class_names):
            if name == "陌生人":
                self.class_thresholds[i] = 0.55
            else:
                self.class_thresholds[i] = 0.65

        # 确保使用TensorFlow 1.x兼容模式
        tf.disable_eager_execution()

        # 查找模型checkpoint
        checkpoint_path = self._find_latest_model()
        if checkpoint_path is None or checkpoint_path == "":
            print("错误: 未找到有效的模型checkpoint路径")
            return False

        print(f"找到模型checkpoint: {checkpoint_path}")

        # 检查checkpoint文件是否存在
        checkpoint_files = [
            checkpoint_path + '.meta',
            checkpoint_path + '.index',
            checkpoint_path + '.data-00000-of-00001'
        ]

        for file_path in checkpoint_files:
            if not os.path.exists(file_path):
                print(f"警告: 模型文件不存在: {file_path}")

        try:
            # 关闭现有会话（如果存在）
            if self.sess is not None:
                try:
                    self.sess.close()
                except:
                    pass
                self.sess = None

            # 重置TensorFlow图
            tf.reset_default_graph()

            # 定义占位符
            size = 64
            self.input_image = tf.placeholder(tf.float32, [None, size, size, 3], name='input_image')
            self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
            self.dropout_rate_2 = tf.placeholder(tf.float32, name='dropout_rate_2')

            # 构建网络
            self.outdata = face_net_tool.layer_net(self.input_image, self.num_classes,
                                     self.dropout_rate, self.dropout_rate_2)

            # 创建会话
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # 创建Saver并恢复模型
            saver = tf.train.Saver()
            print(f"正在恢复模型: {checkpoint_path}")
            saver.restore(self.sess, checkpoint_path)

            print("✅ 模型加载成功")

            # 快速测试模型
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
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()

            # 尝试另一种方法：查找所有检查点文件
            print("尝试查找所有模型文件...")
            model_files = glob.glob(os.path.join(self.model_path, '*.meta'))
            if model_files:
                print(f"找到以下模型文件: {model_files}")
                # 尝试加载第一个找到的模型
                for model_file in model_files:
                    try:
                        model_path = model_file.replace('.meta', '')
                        print(f"尝试加载: {model_path}")

                        # 重新创建会话和图
                        tf.reset_default_graph()
                        self.sess = tf.Session(config=config)

                        # 重新定义网络结构
                        self.input_image = tf.placeholder(tf.float32, [None, size, size, 3], name='input_image')
                        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
                        self.dropout_rate_2 = tf.placeholder(tf.float32, name='dropout_rate_2')
                        self.outdata = layer_net(self.input_image, self.num_classes,
                                                 self.dropout_rate, self.dropout_rate_2)

                        saver = tf.train.Saver()
                        saver.restore(self.sess, model_path)
                        print(f"✅ 成功加载模型: {model_path}")
                        return True
                    except Exception as inner_e:
                        print(f"加载失败: {inner_e}")

            return False

    def _find_latest_model(self):
        """查找最新的模型文件"""
        if not os.path.exists(self.model_path):
            print(f"模型目录 {self.model_path} 不存在")
            return None

        print(f"搜索模型目录: {self.model_path}")

        # 列出目录中的所有文件
        all_files = os.listdir(self.model_path)
        print(f"目录中的文件: {all_files}")

        # 优先检查checkpoint文件
        checkpoint_file = os.path.join(self.model_path, 'checkpoint')
        if os.path.exists(checkpoint_file):
            print("找到checkpoint文件")
            try:
                with open(checkpoint_file, 'r') as f:
                    content = f.read()
                    print(f"checkpoint内容:\n{content}")

                    # 解析checkpoint文件
                    for line in content.split('\n'):
                        if 'model_checkpoint_path' in line and ':' in line:
                            model_name = line.split(':')[1].strip().strip('"')
                            model_path = os.path.join(self.model_path, model_name)
                            print(f"从checkpoint解析出的模型路径: {model_path}")

                            if os.path.exists(model_path + '.meta'):
                                print(f"找到模型文件: {model_path}")
                                return model_path
                            else:
                                print(f"警告: 模型文件 {model_path}.meta 不存在")
            except Exception as e:
                print(f"读取checkpoint文件失败: {e}")

        # 查找best_model
        best_model_patterns = [
            os.path.join(self.model_path, 'best_model-*'),
            os.path.join(self.model_path, 'best_model')
        ]

        for pattern in best_model_patterns:
            model_files = glob.glob(pattern + '.meta')
            if model_files:
                model_files.sort(key=os.path.getmtime, reverse=True)
                latest_model = model_files[0]
                model_path = latest_model.replace('.meta', '')
                print(f"找到best_model: {model_path}")

                required_files = [model_path + ext for ext in ['.meta', '.index', '.data-00000-of-00001']]
                missing_files = [f for f in required_files if not os.path.exists(f)]

                if missing_files:
                    print(f"警告: 缺少文件: {missing_files}")
                else:
                    return model_path

        # 查找final_model
        final_model_path = os.path.join(self.model_path, 'final_model')
        if os.path.exists(final_model_path + '.meta'):
            print(f"找到final_model: {final_model_path}")

            required_files = [final_model_path + ext for ext in ['.meta', '.index', '.data-00000-of-00001']]
            missing_files = [f for f in required_files if not os.path.exists(f)]

            if missing_files:
                print(f"警告: 缺少文件: {missing_files}")
            else:
                return final_model_path

        # 查找任何以.meta结尾的文件
        all_meta_files = glob.glob(os.path.join(self.model_path, '*.meta'))
        if all_meta_files:
            all_meta_files.sort(key=os.path.getmtime, reverse=True)
            latest_meta = all_meta_files[0]
            model_path = latest_meta.replace('.meta', '')
            print(f"找到最近的meta文件: {model_path}")
            return model_path

        print("错误: 未找到任何有效的模型文件")
        return None

    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        if face_img is None or face_img.size == 0:
            return None

        # 调整大小到64x64
        face_resized = cv2.resize(face_img, (64, 64))

        # 直方图均衡化（增强对比度）
        img_yuv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
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
                                       self.dropout_rate: 1.0,
                                       self.dropout_rate_2: 1.0
                                   })

            # 计算softmax概率
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            probs = probs[0]

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
            recent_predictions = list(self.prediction_history)[-5:]
            pred_counter = Counter(recent_predictions)
            most_common_pred, most_common_count = pred_counter.most_common(1)[0]

            if most_common_count >= 3 and most_common_pred != raw_predicted:
                final_predicted = most_common_pred
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

    def _draw_chinese_text(self, frame, text, position, color=(0, 255, 0)):
        """
        使用PIL绘制中文文本到OpenCV图像

        Args:
            frame: OpenCV图像 (BGR格式)
            text: 要绘制的文本
            position: (x, y) 文本位置
            color: (R, G, B) 文本颜色

        Returns:
            ndarray: 绘制后的图像
        """
        try:
            # 将OpenCV图像从BGR转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 创建PIL图像
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)

            # 尝试加载字体
            try:
                # 尝试使用系统中文字体
                font = ImageFont.truetype("simhei.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("Arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

            # 绘制文本
            draw.text(position, text, font=font, fill=color)

            # 将PIL图像转回OpenCV格式
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return frame_bgr
        except Exception as e:
            print(f"绘制中文文本失败: {e}")
            # 如果绘制失败，返回原图
            return frame

    def process_recognition(self, frame, faces):
        """
        处理视频帧进行人脸识别

        Args:
            frame: 视频帧
            faces: 检测到的人脸列表

        Returns:
            tuple: (处理后的帧, 识别结果列表)
        """
        recognition_results = []

        for i, d in enumerate(faces):
            x1 = max(d.top(), 0)
            y1 = min(d.bottom(), frame.shape[0])
            x2 = max(d.left(), 0)
            y2 = min(d.right(), frame.shape[1])

            face_img = frame[x1:y1, x2:y2]
            if face_img.size > 0 and face_img.shape[0] > 20 and face_img.shape[1] > 20:
                # 识别人脸
                predicted_class_idx, confidence, all_probs = self.recognize_with_smoothing(face_img)

                # 安全检查
                if predicted_class_idx is not None:
                    # 获取类别名称
                    if predicted_class_idx < len(self.class_names):
                        person_name = self.class_names[predicted_class_idx]
                    else:
                        person_name = "未知"

                    # 确定显示颜色
                    if confidence > 0.55:
                        if person_name == "陌生人":
                            box_color = (0, 0, 255)  # 红色框
                            text_color = (0, 0, 255)  # 红色文字
                            label = f"陌生人 ({confidence:.2f})"
                        else:
                            box_color = (0, 255, 0)  # 绿色框
                            text_color = (0, 255, 0)  # 绿色文字
                            label = f"{person_name} ({confidence:.2f})"
                    else:
                        box_color = (128, 128, 128)  # 灰色框
                        text_color = (128, 128, 128)  # 灰色文字
                        label = "低置信度"

                    # 绘制识别框（使用对应的颜色）
                    cv2.rectangle(frame, (x2, x1), (y2, y1), box_color, 2)

                    # 使用PIL绘制中文文本（解决乱码问题）
                    frame = self._draw_chinese_text(frame, label, (x2, y1 - 25), text_color)

                    # 记录识别结果
                    recognition_results.append({
                        'name': person_name,
                        'confidence': confidence,
                        'box': (x2, x1, y2, y1),
                        'box_color': box_color,
                        'text_color': text_color
                    })

                    # 触发回调
                    if self.on_recognition_result:
                        self.on_recognition_result(person_name, confidence)
                else:
                    # 如果识别失败
                    box_color = (128, 128, 128)  # 灰色框
                    text_color = (128, 128, 128)  # 灰色文字
                    cv2.rectangle(frame, (x2, x1), (y2, y1), box_color, 2)
                    # 使用PIL绘制中文文本
                    frame = self._draw_chinese_text(frame, "识别失败", (x2, y1 - 25), text_color)

        return frame, recognition_results

    def close(self):
        """关闭资源"""
        if self.sess:
            self.sess.close()


# 人脸检测器类（独立功能）
class FaceDetector:
    """人脸检测器类"""

    def __init__(self):
        try:
            self.detector = dlib.get_frontal_face_detector()
        except Exception as e:
            print(f"无法加载dlib人脸检测器: {e}")
            raise

    def detect_faces(self, frame):
        """
        检测人脸

        Args:
            frame: 视频帧

        Returns:
            list: 检测到的人脸列表
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = self.detector(gray, 1)
        return faces

    def draw_faces(self, frame, faces, color=(255, 255, 0), thickness=2):
        """
        绘制人脸框（普通模式）

        Args:
            frame: 视频帧
            faces: 人脸列表
            color: 框颜色
            thickness: 线宽

        Returns:
            ndarray: 绘制后的帧
        """
        for i, d in enumerate(faces):
            x1 = max(d.top(), 0)
            y1 = min(d.bottom(), frame.shape[0])
            x2 = max(d.left(), 0)
            y2 = min(d.right(), frame.shape[1])
            cv2.rectangle(frame, (x2, x1), (y2, y1), color, thickness)
        return frame
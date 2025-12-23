import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import dlib
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # 禁用TensorFlow 2.x行为
import os
import time
from PIL import Image, ImageTk
import glob
from collections import deque, Counter
import random
from datetime import datetime
import threading
import shutil


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
        beta = random.randint(-30, 30)  # 亮度
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        augmentations.append(bright)

        # 3. 对比度调整
        contrast = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=0)
        augmentations.append(contrast)

        return augmentations

    def capture_data(self, person_name, target_count=100, cap=None):
        """
        采集指定人员的人脸数据
        :param person_name: 人员名称（英文或拼音，不要用中文）
        :param target_count: 目标采集数量（原始+增强后的总数量）
        :param cap: 摄像头对象
        """
        # 创建保存目录：faces_ok/人员名称/
        save_dir = os.path.join('./faces_ok', person_name)

        # 如果目录已存在，先删除所有图片
        if os.path.exists(save_dir):
            for file in os.listdir(save_dir):
                file_path = os.path.join(save_dir, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    os.remove(file_path)
            print(f"已清空 {person_name} 的旧照片")
        else:
            os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"开始采集 [{person_name}] 的人脸数据")
        print(f"目标数量: {target_count}张（含3倍增强）")
        print(f"保存目录: {save_dir}")
        print(f"{'=' * 60}")

        # 检查已有图片数量（应该为0）
        existing_files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))]
        saved_count = len(existing_files)
        print(f"📁 当前目录图片数: {saved_count} 张")

        frame_skip = 1  # 每1帧采集一次，提高采集速度
        frame_counter = 0

        return target_count, save_dir, frame_skip, frame_counter


def train_model():
    """训练模型的函数（从faces_train_multi_person.py提取）"""
    # 检查是否有足够的数据
    faces_ok_dir = './faces_ok'
    if not os.path.exists(faces_ok_dir):
        print("faces_ok目录不存在")
        return False

    class_names = [d for d in os.listdir(faces_ok_dir) if os.path.isdir(os.path.join(faces_ok_dir, d))]
    if len(class_names) < 2:
        print("数据不足，至少需要2个类别")
        return False

    # 添加陌生人类别
    class_names.append("陌生人")
    num_classes = len(class_names)

    print(f"检测到 {len(class_names) - 1} 个人脸类别: {class_names[:-1]}")

    # 读取数据
    def read_data():
        images = []
        labels = []

        for i, class_name in enumerate(class_names):
            if class_name == "陌生人":
                continue  # 陌生人是虚拟类别，不读取实际图片

            class_dir = os.path.join(faces_ok_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        img = img.astype(np.float32) / 255.0
                        images.append(img)

                        # 创建one-hot标签
                        label = [0] * num_classes
                        label[i] = 1
                        labels.append(label)

        return np.array(images), np.array(labels)

    try:
        # 清空模型目录
        model_dir = './model_multi_class'
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        # 读取数据
        X, y = read_data()
        print(f"读取到 {len(X)} 张图片")

        if len(X) == 0:
            print("没有读取到任何图片数据")
            return False

        # 检查数据是否平衡
        class_counts = []
        for i in range(num_classes - 1):  # 不包括陌生人
            count = sum(1 for label in y if np.argmax(label) == i)
            class_counts.append(count)
            print(f"类别 {class_names[i]}: {count} 张图片")

        if min(class_counts) == 0:
            print("某些类别没有数据，无法训练")
            return False

        # 打乱数据
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # 分割训练集和测试集
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

        # 定义模型
        input_image = tf.placeholder(tf.float32, [None, 64, 64, 3], name='input_image')
        input_label = tf.placeholder(tf.float32, [None, num_classes], name='input_label')
        dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
        dropout_rate_2 = tf.placeholder(tf.float32, name='dropout_rate_2')

        outdata = layer_net(input_image, num_classes, dropout_rate, dropout_rate_2)

        # 定义损失函数和优化器
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_label, logits=outdata))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # 定义准确率
        correct_prediction = tf.equal(tf.argmax(outdata, 1), tf.argmax(input_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 保存模型
        saver = tf.train.Saver()

        # 创建会话
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        # 训练模型
        batch_size = 32
        epochs = 50  # 减少训练轮数以加快训练速度

        best_accuracy = 0
        patience = 10
        patience_counter = 0

        print("开始训练模型...")

        for epoch in range(epochs):
            # 计算批次数量
            num_batches = len(X_train) // batch_size

            # 训练
            total_loss = 0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                batch_x = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]

                _, loss_val = sess.run([optimizer, loss],
                                       feed_dict={
                                           input_image: batch_x,
                                           input_label: batch_y,
                                           dropout_rate: 0.5,
                                           dropout_rate_2: 0.5
                                       })
                total_loss += loss_val

            # 测试准确率
            if len(X_test) > 0:
                test_accuracy = sess.run(accuracy,
                                         feed_dict={
                                             input_image: X_test,
                                             input_label: y_test,
                                             dropout_rate: 1.0,
                                             dropout_rate_2: 1.0
                                         })

                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}, Test Accuracy: {test_accuracy:.4f}")

                # 保存最佳模型
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    model_path = f'./model_multi_class/best_model-{epoch + 1}'
                    saver.save(sess, model_path)
                    print(f"保存最佳模型: {model_path}")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("早停，训练结束")
                    break
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}")

        # 保存最终模型
        final_model_path = './model_multi_class/final_model'
        saver.save(sess, final_model_path)

        # 保存类别名称
        with open('./model_multi_class/class_names.txt', 'w', encoding='utf-8') as f:
            for name in class_names:
                f.write(name + '\n')

        print(f"模型训练完成，最佳准确率: {best_accuracy:.4f}")
        sess.close()
        return True

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        self.confidence_history = deque(maxlen=15)  # 保存最近15次置信度

        # 动态阈值参数
        self.base_threshold = 0.65  # 基础置信度阈值
        self.class_thresholds = {}  # 每个类别的动态阈值

    def load_model(self):
        """加载训练好的平衡模型 - 修复版"""
        print(f"正在从 {self.model_path} 加载模型...")

        # 检查模型目录是否存在
        if not os.path.exists(self.model_path):
            print(f"错误: 模型目录 {self.model_path} 不存在")
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

        # 确保使用TensorFlow 1.x兼容模式
        tf.disable_eager_execution()

        # 查找模型checkpoint
        checkpoint_path = self.find_latest_model()
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
            self.outdata = layer_net(self.input_image, self.num_classes,
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

    def find_latest_model(self):
        """查找最新的模型文件"""
        model_dir = './model_multi_class/'
        if not os.path.exists(model_dir):
            print(f"模型目录 {model_dir} 不存在")
            return None

        print(f"搜索模型目录: {model_dir}")

        # 列出目录中的所有文件
        all_files = os.listdir(model_dir)
        print(f"目录中的文件: {all_files}")

        # 优先检查checkpoint文件
        checkpoint_file = os.path.join(model_dir, 'checkpoint')
        if os.path.exists(checkpoint_file):
            print("找到checkpoint文件")
            try:
                with open(checkpoint_file, 'r') as f:
                    content = f.read()
                    print(f"checkpoint内容:\n{content}")

                    # 解析checkpoint文件
                    for line in content.split('\n'):
                        if 'model_checkpoint_path' in line and ':' in line:
                            # 提取模型名称，格式如: model_checkpoint_path: "best_model-50"
                            model_name = line.split(':')[1].strip().strip('"')
                            model_path = os.path.join(model_dir, model_name)
                            print(f"从checkpoint解析出的模型路径: {model_path}")

                            # 检查模型文件是否存在
                            if os.path.exists(model_path + '.meta'):
                                print(f"找到模型文件: {model_path}")
                                return model_path
                            else:
                                print(f"警告: 模型文件 {model_path}.meta 不存在")
            except Exception as e:
                print(f"读取checkpoint文件失败: {e}")

        # 如果checkpoint不存在或解析失败，查找best_model
        best_model_patterns = [
            os.path.join(model_dir, 'best_model-*'),  # 旧格式
            os.path.join(model_dir, 'best_model')  # 新格式
        ]

        for pattern in best_model_patterns:
            model_files = glob.glob(pattern + '.meta')
            if model_files:
                # 按修改时间排序，取最新的
                model_files.sort(key=os.path.getmtime, reverse=True)
                latest_model = model_files[0]
                model_path = latest_model.replace('.meta', '')
                print(f"找到best_model: {model_path}")

                # 检查其他必要的文件
                required_files = [model_path + ext for ext in ['.meta', '.index', '.data-00000-of-00001']]
                missing_files = [f for f in required_files if not os.path.exists(f)]

                if missing_files:
                    print(f"警告: 缺少文件: {missing_files}")
                else:
                    return model_path

        # 查找final_model
        final_model_path = os.path.join(model_dir, 'final_model')
        if os.path.exists(final_model_path + '.meta'):
            print(f"找到final_model: {final_model_path}")

            # 检查其他必要的文件
            required_files = [final_model_path + ext for ext in ['.meta', '.index', '.data-00000-of-00001']]
            missing_files = [f for f in required_files if not os.path.exists(f)]

            if missing_files:
                print(f"警告: 缺少文件: {missing_files}")
            else:
                return final_model_path

        # 查找任何以.meta结尾的文件
        all_meta_files = glob.glob(os.path.join(model_dir, '*.meta'))
        if all_meta_files:
            # 按修改时间排序，取最新的
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

    def close(self):
        """关闭资源"""
        if self.sess:
            self.sess.close()


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统 - 完整修复版 - 模型加载修复 - 自动重新识别 - 立即重启")
        self.root.geometry("1200x800")  # 增大窗口

        # 初始化变量
        self.cap = None
        self.is_running = False
        self.is_collecting = False
        self.collection_count = 0
        self.max_collection = 500  # 默认采集500张
        self.current_user = "default_user"
        self.camera_index = 0  # 摄像头索引
        self.target_count = 0
        self.save_dir = ""
        self.frame_skip = 1  # 提高采集速度
        self.frame_counter = 0

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
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.is_running = True
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

        # 采集数量输入
        count_frame = ttk.Frame(control_frame)
        count_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(count_frame, text="采集数量:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.count_entry = ttk.Entry(count_frame, width=15, font=("Arial", 9))
        self.count_entry.insert(0, str(self.max_collection))
        self.count_entry.pack(side=tk.RIGHT)

        # 摄像头选择
        camera_frame = ttk.Frame(control_frame)
        camera_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(camera_frame, text="摄像头:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="内置")
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, state="readonly", width=12)
        camera_combo['values'] = ("内置(0)", "外接(1)")
        camera_combo.pack(side=tk.RIGHT)

        # 按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        self.collect_btn = ttk.Button(btn_frame, text="人脸采集", command=self.toggle_collection, width=10)
        self.collect_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.recognize_btn = ttk.Button(btn_frame, text="人脸识别", command=self.toggle_recognition, width=10)
        self.recognize_btn.pack(side=tk.LEFT, padx=(0, 5))
        if not self.model_loaded:
            self.recognize_btn.config(state=tk.DISABLED)

        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop_all, state=tk.NORMAL, width=10)
        self.stop_btn.pack(side=tk.RIGHT)

        self.status_label = ttk.Label(control_frame, text="状态: 摄像头已启动", font=("Arial", 9))
        self.status_label.pack(pady=(5, 0))

        # 进度条
        self.progress = ttk.Progressbar(control_frame, mode='determinate', length=200)
        self.progress['maximum'] = self.max_collection
        self.progress.pack(pady=(5, 0), fill=tk.X)
        self.progress.pack_forget()  # 默认隐藏

        # 训练进度条
        self.train_progress = ttk.Progressbar(control_frame, mode='indeterminate', length=200)
        self.train_progress.pack(pady=(5, 0), fill=tk.X)
        self.train_progress.pack_forget()  # 默认隐藏

        # 日志信息区域
        log_frame = ttk.LabelFrame(right_frame, text="日志信息", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.info_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Arial", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def toggle_collection(self):
        if not self.is_running:
            messagebox.showwarning("警告", "摄像头未启动")
            return

        if not self.is_collecting:
            self.start_collection()
        else:
            self.stop_collection()

    def toggle_recognition(self):
        if not self.is_running:
            messagebox.showwarning("警告", "摄像头未启动")
            return

        if self.is_collecting:
            self.stop_collection()

        # 切换识别状态
        if self.is_running and not self.is_collecting and self.model_loaded:
            self.is_running = False
            self.recognize_btn.config(text="人脸识别")
            self.status_label.config(text="状态: 摄像头已停止识别")
            self.update_info("人脸识别已停止")
        else:
            if not self.model_loaded:
                messagebox.showwarning("警告", "模型未加载，请先训练模型")
                return
            self.is_running = True
            self.recognize_btn.config(text="停止识别")
            self.status_label.config(text="状态: 正在识别人脸")
            self.update_info("开始人脸识别...")
            self.update_video()

    def start_collection(self):
        self.current_user = self.user_entry.get().strip()
        if not self.current_user:
            self.current_user = "default_user"

        # 获取采集数量
        try:
            count = int(self.count_entry.get())
            if count < 20:
                count = 20
            self.max_collection = count
        except:
            self.max_collection = 500  # 默认500张

        # 获取摄像头索引
        if self.camera_var.get() == "外接(1)":
            self.camera_index = 1
        else:
            self.camera_index = 0

        # 重新打开摄像头（如果需要切换摄像头）
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        # 初始化采集参数
        try:
            self.target_count, self.save_dir, self.frame_skip, self.frame_counter = self.face_collector.capture_data(
                self.current_user, self.max_collection)
        except Exception as e:
            messagebox.showerror("错误", f"初始化采集失败: {e}")
            return

        self.is_collecting = True
        self.collection_count = 0
        # 检查已有图片数量
        existing_files = [f for f in os.listdir(self.save_dir) if f.endswith(('.jpg', '.png'))]
        self.collection_count = len(existing_files)

        self.collect_btn.config(text="停止采集")
        self.recognize_btn.config(state=tk.DISABLED)
        self.progress.pack(pady=(5, 0), fill=tk.X)  # 显示进度条
        self.progress['maximum'] = self.target_count
        self.progress['value'] = self.collection_count
        self.status_label.config(text=f"状态: 正在采集人脸 ({self.collection_count}/{self.target_count})")
        self.update_info(f"开始采集用户 '{self.current_user}' 的人脸数据...")

    def stop_collection(self):
        self.is_collecting = False
        self.collect_btn.config(text="人脸采集")
        self.recognize_btn.config(state=tk.NORMAL if self.model_loaded else tk.DISABLED)
        self.progress.pack_forget()  # 隐藏进度条
        self.status_label.config(text="状态: 摄像头已停止采集")
        self.update_info(f"人脸采集停止，共保存 {self.collection_count} 张图片")

    def start_training(self):
        """开始模型训练"""
        # 检查是否有足够的数据
        faces_ok_dir = './faces_ok'
        if not os.path.exists(faces_ok_dir):
            messagebox.showerror("错误", "faces_ok目录不存在")
            return

        class_names = [d for d in os.listdir(faces_ok_dir) if os.path.isdir(os.path.join(faces_ok_dir, d))]
        if len(class_names) < 2:
            messagebox.showerror("错误", "至少需要2个类别才能训练模型")
            return

        # 检查当前是否有采集或识别在运行
        if self.is_collecting:
            self.stop_collection()
        if self.is_running:
            self.is_running = False
            self.recognize_btn.config(text="人脸识别")

        # 显示训练进度条
        self.train_progress.pack(pady=(5, 0), fill=tk.X)
        self.train_progress.start(10)
        self.status_label.config(text="状态: 正在训练模型...")
        self.update_info("开始训练模型...")

        # 禁用相关按钮
        self.collect_btn.config(state=tk.DISABLED)
        self.recognize_btn.config(state=tk.DISABLED)

        # 在新线程中训练模型
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        """训练模型的线程函数"""
        try:
            success = train_model()
            self.root.after(0, self._training_complete, success)
        except Exception as e:
            print(f"训练线程出错: {e}")
            self.root.after(0, self._training_complete, False)

    def _training_complete(self, success):
        """训练完成后的回调"""
        # 停止并隐藏训练进度条
        self.train_progress.stop()
        self.train_progress.pack_forget()

        # 重新启用按钮
        self.collect_btn.config(state=tk.NORMAL)

        if success:
            self.update_info("模型训练完成")
            self.status_label.config(text="状态: 模型训练完成")

            # 关键修复：先关闭旧的识别器会话，重置TensorFlow图
            if self.face_recognizer:
                try:
                    self.face_recognizer.close()
                    self.face_recognizer.sess = None
                except:
                    pass  # 如果关闭失败，忽略

            # 重置TensorFlow的默认图
            tf.reset_default_graph()

            # 创建新的识别器实例
            self.face_recognizer = ImprovedFaceRecognizer()

            # 尝试多次加载模型，每次之间有延迟
            max_retries = 3
            for retry in range(max_retries):
                self.model_loaded = self.face_recognizer.load_model()
                if self.model_loaded:
                    break
                if retry < max_retries - 1:
                    self.update_info(f"模型加载失败，正在重试 ({retry + 1}/{max_retries})...")
                    time.sleep(1)  # 等待1秒再重试

            # 更新模型状态显示
            model_status_frame = self.root.nametowidget(
                self.root.winfo_children()[0].winfo_children()[1].winfo_children()[1])
            for widget in model_status_frame.winfo_children():
                widget.destroy()

            if self.model_loaded:
                status_text = f"✓ 模型加载成功!\n- 类别数: {self.face_recognizer.num_classes}\n- 类别名称: {self.face_recognizer.class_names}"
                ttk.Label(model_status_frame, text=status_text, foreground="green", font=("Arial", 9)).pack(anchor=tk.W)
                self.recognize_btn.config(state=tk.NORMAL)

                # 立即开始人脸识别
                self.is_running = True
                self.recognize_btn.config(text="停止识别")
                self.status_label.config(text="状态: 正在识别人脸")
                self.update_info("训练完成，立即开始人脸识别...")

                # 立即开始更新视频（触发识别）
                self.update_video()
            else:
                status_text = "✗ 模型加载失败\n💡 请尝试重新启动程序"
                ttk.Label(model_status_frame, text=status_text, foreground="red", font=("Arial", 9)).pack(anchor=tk.W)
                self.recognize_btn.config(state=tk.DISABLED)
                self.update_info("模型加载失败，请重新启动程序")
        else:
            self.update_info("模型训练失败")
            self.status_label.config(text="状态: 模型训练失败")
            messagebox.showerror("错误", "模型训练失败，请检查faces_ok目录中的数据")

            # 重新启用识别按钮
            self.recognize_btn.config(state=tk.NORMAL if self.model_loaded else tk.DISABLED)

    def stop_all(self):
        """停止所有操作但保持摄像头运行"""
        self.is_running = False
        self.is_collecting = False
        self.recognize_btn.config(text="人脸识别", state=tk.NORMAL if self.model_loaded else tk.DISABLED)
        self.collect_btn.config(text="人脸采集", state=tk.NORMAL)
        self.progress.pack_forget()  # 隐藏进度条
        self.train_progress.stop()
        self.train_progress.pack_forget()  # 隐藏训练进度条
        self.status_label.config(text="状态: 摄像头已停止")
        self.update_info("摄像头已停止")

    def update_video(self):
        if not self.cap or not self.cap.isOpened():
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

            if self.is_collecting:
                # 采集模式：检测到人脸时保存照片
                face_img = frame[x1:y1, x2:y2]
                if face_img.size > 0 and face_img.shape[0] > 20 and face_img.shape[1] > 20:  # 确保人脸足够大
                    self.frame_counter += 1

                    # 每frame_skip帧采集一次
                    if self.frame_counter % self.frame_skip == 0 and self.collection_count < self.target_count:
                        face_resized = cv2.resize(face_img, (64, 64))

                        # 生成时间戳作为文件名
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        base_filename = f"{self.current_user}_{timestamp}"

                        # 保存原始图片
                        original_path = os.path.join(self.save_dir, f"{base_filename}_orig.jpg")
                        cv2.imwrite(original_path, face_resized)
                        self.collection_count += 1

                        # 自动生成3个增强版本
                        if self.collection_count < self.target_count:
                            augmentations = self.face_collector.apply_augmentations(face_resized)
                            for i, aug_img in enumerate(augmentations):
                                if self.collection_count >= self.target_count:
                                    break

                                aug_path = os.path.join(self.save_dir, f"{base_filename}_aug{i + 1}.jpg")
                                cv2.imwrite(aug_path, aug_img)
                                self.collection_count += 1

                        # 更新进度条和状态
                        self.progress['value'] = self.collection_count
                        self.status_label.config(
                            text=f"状态: 正在采集人脸 ({self.collection_count}/{self.target_count})")

                        if self.collection_count >= self.target_count:
                            self.stop_collection()
                            # 自动开始训练模型
                            self.start_training()
                            break

                # 绘制绿色采集框
                cv2.rectangle(frame, (x2, x1), (y2, y1), (0, 255, 0), 2)  # 绿色框

                # 使用OpenCV绘制中文文本（解决乱码问题）
                # 先创建一个PIL图像，绘制中文文本，然后转回OpenCV格式
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img_pil)
                try:
                    # 尝试使用系统字体
                    font = ImageFont.truetype("simhei.ttf", 20)
                except:
                    try:
                        font = ImageFont.truetype("Arial.ttf", 20)
                    except:
                        font = ImageFont.load_default()

                draw.text((x2, y1 - 25), f"采集中: {self.collection_count}/{self.target_count}", font=font,
                          fill=(0, 255, 0))
                frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            elif self.is_running and not self.is_collecting and self.model_loaded:
                # 识别模式：识别人脸
                face_img = frame[x1:y1, x2:y2]
                if face_img.size > 0 and face_img.shape[0] > 20 and face_img.shape[1] > 20:  # 确保人脸足够大
                    # 识别人脸（使用face_recognition_multi_person.py的改进识别）
                    predicted_class_idx, confidence, all_probs = self.face_recognizer.recognize_with_smoothing(face_img)

                    # 安全检查：如果识别结果为None，跳过
                    if predicted_class_idx is not None:
                        # 获取类别名称
                        if predicted_class_idx < len(self.face_recognizer.class_names):
                            person_name = self.face_recognizer.class_names[predicted_class_idx]
                        else:
                            person_name = "未知"

                        # 根据置信度设置标签
                        if confidence > 0.55:
                            if person_name == "陌生人":
                                color = (0, 0, 255)  # 红色 - 陌生人
                                label = f"陌生人 ({confidence:.2f})"
                            else:
                                color = (0, 255, 0)  # 绿色 - 已知人员
                                label = f"{person_name} ({confidence:.2f})"
                        else:
                            color = (128, 128, 128)  # 灰色 - 置信度低
                            label = "低置信度"

                        # 使用OpenCV绘制中文文本（解决乱码问题）
                        # 先创建一个PIL图像，绘制中文文本，然后转回OpenCV格式
                        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(img_pil)
                        try:
                            # 尝试使用系统字体
                            font = ImageFont.truetype("simhei.ttf", 20)
                        except:
                            try:
                                font = ImageFont.truetype("Arial.ttf", 20)
                            except:
                                font = ImageFont.load_default()

                        draw.text((x2, y1 - 25), label, font=font, fill=color)
                        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                        # 更新信息显示
                        info_msg = f"识别结果: {person_name}, 置信度: {confidence:.3f}"
                        self.update_info(info_msg)
                    else:
                        # 如果识别失败，显示"识别失败"
                        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(img_pil)
                        try:
                            # 尝试使用系统字体
                            font = ImageFont.truetype("simhei.ttf", 20)
                        except:
                            try:
                                font = ImageFont.truetype("Arial.ttf", 20)
                            except:
                                font = ImageFont.load_default()

                        draw.text((x2, y1 - 25), "识别失败", font=font, fill=(128, 128, 128))
                        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # 绘制蓝色识别框
                cv2.rectangle(frame, (x2, x1), (y2, y1), (255, 0, 0), 2)  # 蓝色框
            else:
                # 普通显示模式（摄像头开启但不进行识别或采集）
                cv2.rectangle(frame, (x2, x1), (y2, y1), (255, 255, 0), 2)  # 黄色框

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
        # 关闭摄像头
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # 关闭识别器
        if hasattr(self, 'face_recognizer'):
            self.face_recognizer.close()
        # 关闭采集器
        if hasattr(self, 'face_collector'):
            # 如果有正在运行的采集，需要处理
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()


if __name__ == "__main__":
    main()




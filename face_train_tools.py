"""
faces_train_multi_person.py - 人脸识别模型训练工具类
重构为可调用的模块，支持多类别人脸识别训练
"""

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil
import random
import os
import time
import matplotlib.pyplot as plt
import sys
import threading

# 启用v1兼容模式
tf.compat.v1.disable_eager_execution()


def layer_net_tf1(input_image, num_class, dropout_rate, dropout_rate_2):
    """
    TensorFlow 1.x 版本的layer_net函数
    与主窗口中的layer_net函数保持一致
    """
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

    """第八层，输出层，输入1*512，输出1*num_class"""
    w5 = tf.Variable(tf.random.normal([512, num_class]), name='w5')  # 输入通道(512)， 输出通道(num_class)
    b5 = tf.Variable(tf.random.normal([num_class]), name='b5')
    outdata = tf.add(tf.matmul(drop4, w5), b5)  # (1,512)*(512,num_class)=(1,num_class)

    return outdata


class BalancedDataLoader:
    """平衡数据加载器，解决类别不平衡问题"""

    def __init__(self, target_samples_per_class=500):
        self.target_samples = target_samples_per_class
        self.augmentation_techniques = [
            self.horizontal_flip,
            self.adjust_brightness,
            self.adjust_contrast,
            self.random_rotation,
            self.add_noise
        ]

    def horizontal_flip(self, img):
        """水平翻转"""
        return cv2.flip(img, 1)

    def adjust_brightness(self, img):
        """调整亮度"""
        alpha = random.uniform(0.7, 1.3)
        beta = random.randint(-30, 30)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def adjust_contrast(self, img):
        """调整对比度"""
        alpha = random.uniform(0.8, 1.2)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    def random_rotation(self, img):
        """随机小角度旋转"""
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def add_noise(self, img):
        """添加高斯噪声"""
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, noise)
        return np.clip(noisy_img, 0, 255)

    def augment_image(self, img, num_augment=3):
        """增强单张图片"""
        augmented = [img]
        for _ in range(num_augment):
            # 随机选择增强技术
            technique = random.choice(self.augmentation_techniques)
            try:
                aug_img = technique(img.copy())
                augmented.append(aug_img)
            except:
                # 如果增强失败，使用原图
                augmented.append(img.copy())
        return augmented

    def load_balanced_data(self, faces_ok_dir, faces_no_dir, size=64):
        """加载并平衡数据"""
        imgs = []
        labs = []
        class_names = []

        # 获取faces_ok下的所有人员目录
        person_dirs = []
        for item in os.listdir(faces_ok_dir):
            item_path = os.path.join(faces_ok_dir, item)
            if os.path.isdir(item_path):
                person_dirs.append((item, item_path))
                class_names.append(item)

        if not person_dirs:
            print(f"错误: {faces_ok_dir} 中没有找到人员目录")
            return None, None, None

        # 添加陌生人类别
        class_names.append("陌生人")
        num_classes = len(class_names)

        print(f"📊 数据统计:")
        print(f"   已知人员: {len(person_dirs)} 人")
        print(f"   总类别数: {num_classes}")

        # 第一步：统计每个类别的原始图片数量
        class_counts = {}
        for person_name, person_path in person_dirs:
            files = [f for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[person_name] = len(files)
            print(f"   {person_name}: {len(files)} 张原始图片")

        # 统计陌生人图片数量
        stranger_files = []
        if os.path.exists(faces_no_dir):
            stranger_files = [f for f in os.listdir(faces_no_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts["陌生人"] = len(stranger_files)
        print(f"   陌生人: {len(stranger_files)} 张原始图片")

        # 第二步：确定每个类别的目标样本数
        # 使用最小类别的2倍作为上限，确保平衡
        min_count = min(class_counts.values())
        target_per_class = min(self.target_samples, min_count * 2)
        print(f"\n⚖️ 平衡策略:")
        print(f"   最小类别样本数: {min_count}")
        print(f"   每类目标样本数: {target_per_class}")

        # 第三步：加载并平衡已知人员数据
        for idx, (person_name, person_path) in enumerate(person_dirs):
            print(f"\n📥 加载 {person_name} 的数据...")

            # 获取该人员的所有图片
            files = [f for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # 如果图片太多，随机选择一部分
            if len(files) > target_per_class:
                files = random.sample(files, target_per_class)

            loaded = 0
            need_augment = target_per_class - len(files)

            for file_idx, filename in enumerate(files):
                img_path = os.path.join(person_path, filename)
                img = cv2.imread(img_path)

                if img is not None:
                    # 统一尺寸
                    img_resized = cv2.resize(img, (size, size))

                    # 添加原始图片
                    imgs.append(img_resized)

                    # 创建one-hot标签
                    label = [0] * num_classes
                    label[idx] = 1
                    labs.append(label)
                    loaded += 1

                    # 如果需要增强且是前几个样本
                    if need_augment > 0 and file_idx < min(20, len(files)):
                        # 为每个原始图片创建2个增强版本
                        augmented_imgs = self.augment_image(img_resized, num_augment=2)
                        for aug_img in augmented_imgs[1:]:  # 跳过原始图片
                            imgs.append(aug_img)
                            labs.append(label.copy())
                            loaded += 1
                            need_augment -= 1
                            if need_augment <= 0:
                                break

                if loaded >= target_per_class:
                    break

            print(f"   ✅ 加载完成: {loaded} 张图片")

        # 第四步：加载并平衡陌生人数据
        print(f"\n📥 加载陌生人数据...")
        if os.path.exists(faces_no_dir) and stranger_files:
            # 如果陌生人图片太多，随机选择
            if len(stranger_files) > target_per_class:
                selected_files = random.sample(stranger_files, target_per_class)
            else:
                selected_files = stranger_files

            loaded = 0
            need_augment = target_per_class - len(selected_files)

            for file_idx, filename in enumerate(selected_files):
                img_path = os.path.join(faces_no_dir, filename)
                img = cv2.imread(img_path)

                if img is not None:
                    # 统一尺寸
                    img_resized = cv2.resize(img, (size, size))

                    # 添加原始图片
                    imgs.append(img_resized)

                    # 创建one-hot标签（最后一个类别）
                    label = [0] * num_classes
                    label[-1] = 1
                    labs.append(label)
                    loaded += 1

                    # 如果需要增强
                    if need_augment > 0 and file_idx < min(20, len(selected_files)):
                        augmented_imgs = self.augment_image(img_resized, num_augment=2)
                        for aug_img in augmented_imgs[1:]:
                            imgs.append(aug_img)
                            labs.append(label.copy())
                            loaded += 1
                            need_augment -= 1
                            if need_augment <= 0:
                                break

                if loaded >= target_per_class:
                    break

            print(f"   ✅ 加载完成: {loaded} 张图片")
        else:
            print(f"   ⚠️ 陌生人目录不存在或为空")
            # 创建一些虚拟的陌生人数据
            for i in range(target_per_class):
                # 创建随机图像作为陌生人数据
                random_img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                imgs.append(random_img)

                label = [0] * num_classes
                label[-1] = 1
                labs.append(label)

            print(f"   ✅ 生成虚拟数据: {target_per_class} 张图片")

        return np.array(imgs), np.array(labs), class_names


class FaceModelTrainer:
    """人脸识别模型训练工具类"""

    def __init__(self,
                 faces_ok_dir='./faces_ok',
                 faces_no_dir='./faces_no',
                 model_dir='./model_multi_class',
                 size=64,
                 batch_size=32,
                 learning_rate=0.001,
                 target_samples_per_class=400,
                 num_epochs=100,
                 patience=10):
        """
        初始化训练器

        Args:
            faces_ok_dir: 已知人脸数据目录
            faces_no_dir: 陌生人数据目录
            model_dir: 模型保存目录
            size: 图像大小
            batch_size: 批次大小
            learning_rate: 学习率
            target_samples_per_class: 每类目标样本数
            num_epochs: 训练轮数
            patience: 早停耐心值
        """
        self.faces_ok_dir = faces_ok_dir
        self.faces_no_dir = faces_no_dir
        self.model_dir = model_dir
        self.size = size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_samples = target_samples_per_class
        self.num_epochs = num_epochs
        self.patience = patience

        # 训练结果
        self.best_val_acc = 0
        self.train_losses = []
        self.val_accuracies = []
        self.class_names = []
        self.num_classes = 0

        # TensorFlow占位符
        self.input_image = None
        self.input_label = None
        self.dropout_rate = None
        self.dropout_rate_2 = None
        self.outdata = None
        self.cross_entropy = None
        self.optimizer = None
        self.accuracy = None
        self.saver = None

    def check_directories(self):
        """检查必要的目录是否存在"""
        print("检查项目目录...")

        if not os.path.exists(self.faces_ok_dir):
            print(f"❌ 错误: {self.faces_ok_dir} 目录不存在")
            return False

        if not os.path.exists(self.faces_no_dir):
            print(f"⚠️ 警告: {self.faces_no_dir} 目录不存在，将生成虚拟陌生人数据")
            os.makedirs(self.faces_no_dir, exist_ok=True)

        # 创建模型目录
        if os.path.exists(self.model_dir):
            print(f"⚠️ 警告: {self.model_dir} 目录已存在，将清空目录")
            shutil.rmtree(self.model_dir)

        os.makedirs(self.model_dir, exist_ok=True)
        print(f"✅ 模型目录创建: {self.model_dir}")

        return True

    def load_and_balance_data(self):
        """加载并平衡数据"""
        print("\n📥 加载数据并平衡...")

        # 创建数据加载器
        data_loader = BalancedDataLoader(target_samples_per_class=self.target_samples)

        # 加载数据
        imgs, labs, class_names = data_loader.load_balanced_data(
            self.faces_ok_dir, self.faces_no_dir, self.size
        )

        if imgs is None:
            print("❌ 数据加载失败")
            return None, None, None, None, None

        self.class_names = class_names
        self.num_classes = len(class_names)

        print(f"\n📊 最终数据统计:")
        print(f"   总图片数: {len(imgs)}")

        # 统计各类别数量
        lab_array = np.array(labs)
        for i in range(self.num_classes):
            count = np.sum(lab_array[:, i])
            print(f"   类别 {i} ({self.class_names[i]}): {count} 张")

        return imgs, labs

    def preprocess_data(self, imgs, labs):
        """预处理数据"""
        print(f"\n🔀 划分训练集和测试集...")

        # 划分训练集和测试集
        train_x, test_x, train_y, test_y = train_test_split(
            imgs, labs, test_size=0.2, random_state=42,
            stratify=np.argmax(labs, axis=1)  # 分层抽样
        )

        print(f"   训练集: {len(train_x)} 张")
        print(f"   测试集: {len(test_x)} 张")

        # 归一化和重塑
        train_x = train_x.astype('float32') / 255.0
        test_x = test_x.astype('float32') / 255.0

        train_x = train_x.reshape(-1, self.size, self.size, 3)
        test_x = test_x.reshape(-1, self.size, self.size, 3)

        return train_x, test_x, train_y, test_y

    def build_model(self):
        """构建模型"""
        print(f"\n🧠 构建神经网络...")

        # 定义TensorFlow占位符
        self.input_image = tf.compat.v1.placeholder(tf.float32, [None, self.size, self.size, 3])
        self.input_label = tf.compat.v1.placeholder(tf.float32, [None, self.num_classes])
        self.dropout_rate = tf.compat.v1.placeholder(tf.float32)
        self.dropout_rate_2 = tf.compat.v1.placeholder(tf.float32)

        # 构建网络
        self.outdata = layer_net_tf1(self.input_image, self.num_classes,
                                     self.dropout_rate, self.dropout_rate_2)

        # 定义损失函数（带类别权重）
        class_weights = tf.constant([1.0] * self.num_classes, dtype=tf.float32)

        # 计算加权交叉熵损失
        unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.input_label, logits=self.outdata
        )

        # 计算每个样本的权重
        sample_weights = tf.reduce_sum(self.input_label * class_weights, axis=1)
        weighted_loss = unweighted_loss * sample_weights
        self.cross_entropy = tf.reduce_mean(weighted_loss)

        # 定义优化器
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

        # 定义准确率计算
        correct_prediction = tf.equal(tf.argmax(self.outdata, 1), tf.argmax(self.input_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 创建Saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=3)

        print(f"✅ 模型构建完成")

    def train_model(self, train_x, train_y, test_x, test_y):
        """训练模型"""
        # 训练参数
        num_batches = len(train_x) // self.batch_size

        # 早停参数
        best_val_acc = 0
        patience_counter = 0

        # 训练记录
        self.train_losses = []
        self.val_accuracies = []

        # 创建TensorFlow配置
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            # 初始化变量
            sess.run(tf.compat.v1.global_variables_initializer())

            print(f"\n🚀 开始训练...")
            print(f"   训练样本: {len(train_x)}")
            print(f"   测试样本: {len(test_x)}")
            print(f"   批次大小: {self.batch_size}")
            print(f"   每轮批次: {num_batches}")
            print(f"   最大轮次: {self.num_epochs}")
            print(f"   早停耐心值: {self.patience}")

            # 训练循环
            for epoch in range(self.num_epochs):
                epoch_losses = []

                # 打乱训练数据
                indices = np.arange(len(train_x))
                np.random.shuffle(indices)
                train_x_shuffled = train_x[indices]
                train_y_shuffled = train_y[indices]

                # 批次训练
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, len(train_x_shuffled))

                    batch_x = train_x_shuffled[start_idx:end_idx]
                    batch_y = train_y_shuffled[start_idx:end_idx]

                    # 训练步骤
                    _, loss = sess.run([self.optimizer, self.cross_entropy],
                                       feed_dict={self.input_image: batch_x,
                                                  self.input_label: batch_y,
                                                  self.dropout_rate: 0.5,
                                                  self.dropout_rate_2: 0.3})

                    epoch_losses.append(loss)

                # 计算平均损失
                avg_loss = np.mean(epoch_losses)
                self.train_losses.append(avg_loss)

                # 计算验证准确率
                val_acc = self.accuracy.eval(feed_dict={
                    self.input_image: test_x,
                    self.input_label: test_y,
                    self.dropout_rate: 1.0,
                    self.dropout_rate_2: 1.0
                })
                self.val_accuracies.append(val_acc)

                # 输出训练进度
                print(f"📍 轮次 {epoch + 1:3d}/{self.num_epochs} - "
                      f"损失: {avg_loss:.4f} - "
                      f"验证准确率: {val_acc:.4f}")

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_val_acc = best_val_acc
                    self.saver.save(sess, os.path.join(self.model_dir, 'best_model'))
                    print(f"   ✅ 保存最佳模型 (准确率: {val_acc:.4f})")
                    patience_counter = 0

                    # 保存损失记录
                    with open(os.path.join(self.model_dir, 'loss.txt'), 'w') as f:
                        for loss_val in self.train_losses:
                            f.write(f"{loss_val}\n")
                else:
                    patience_counter += 1

                # 早停检查
                if patience_counter >= self.patience:
                    print(f"\n🛑 早停触发!")
                    print(f"   连续 {self.patience} 轮验证准确率未提升")
                    break

                # 如果准确率足够高，提前停止
                if val_acc > 0.95:
                    print(f"\n🎯 达到目标准确率!")
                    break

            # 保存最终模型
            self.saver.save(sess, os.path.join(self.model_dir, 'final_model'))

            # 保存类别名称
            with open(os.path.join(self.model_dir, 'class_names.txt'), 'w', encoding='utf-8') as f:
                for name in self.class_names:
                    f.write(name + '\n')

            print(f"\n✅ 训练完成!")
            print(f"   最佳验证准确率: {best_val_acc:.4f}")
            print(f"   最终模型保存到: {self.model_dir}")

            # 绘制训练曲线
            if len(self.train_losses) > 1:
                self.plot_training_history()

            return True

    def plot_training_history(self):
        """绘制训练历史曲线"""
        try:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(self.val_accuracies)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
            plt.close()  # 关闭图像，避免阻塞
            print(f"📊 训练曲线保存到: {os.path.join(self.model_dir, 'training_history.png')}")
        except Exception as e:
            print(f"⚠️ 绘制训练曲线失败: {e}")

    def train(self):
        """完整的训练流程"""
        try:
            print("=" * 60)
            print("          平衡数据人脸识别模型训练")
            print("=" * 60)

            # 1. 检查目录
            if not self.check_directories():
                return False

            # 2. 加载和平衡数据
            imgs, labs = self.load_and_balance_data()
            if imgs is None:
                return False

            # 3. 预处理数据
            train_x, test_x, train_y, test_y = self.preprocess_data(imgs, labs)

            # 4. 构建模型
            self.build_model()

            # 5. 训练模型
            success = self.train_model(train_x, train_y, test_x, test_y)

            if success:
                print(f"\n🎉 所有训练完成!")
                print(f"   模型保存在: {self.model_dir}/")
                print(f"   类别数: {self.num_classes}")
                print(f"   类别名称: {self.class_names}")

            return success

        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False


# 主程序入口（保留用于独立运行）
if __name__ == '__main__':
    """独立运行训练程序"""

    # 创建训练器实例
    trainer = FaceModelTrainer(
        faces_ok_dir='./faces_ok',
        faces_no_dir='./faces_no',
        model_dir='./model_multi_class',
        size=64,
        batch_size=32,
        learning_rate=0.001,
        target_samples_per_class=400,
        num_epochs=100,
        patience=10
    )

    # 开始训练
    success = trainer.train()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)
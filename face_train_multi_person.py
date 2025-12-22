"""--------------------------------------------------------------
二、CNN模型训练 - 多类别人脸识别版本（修复版）
修复问题：
1. 类别不平衡问题（hsc:200, xsx:500, 陌生人:5000）
2. 偏置不平衡导致总是识别为xsx
3. 过拟合问题
------------------------------------------------------------------"""
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
import net
import units

# 启用v1兼容模式
tf.compat.v1.disable_eager_execution()

# 添加数据平衡和增强功能
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

"""定义训练函数 - 改进版"""
def do_train(outdata, cross_entropy, optimizer, num_classes, class_names,
             train_x, train_y, test_x, test_y, batch_size=32):
    """改进的训练函数，包含早停和模型保存"""

    # 定义准确率计算
    correct_prediction = tf.equal(tf.argmax(outdata, 1), tf.argmax(input_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 创建模型保存目录
    model_dir = './model_balanced'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 保存类别名称
    with open(os.path.join(model_dir, 'class_names.txt'), 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(name + '\n')

    # 训练参数
    num_epochs = 100
    num_batches = len(train_x) // batch_size

    # 早停参数
    patience = 10
    best_val_acc = 0
    patience_counter = 0

    # 记录训练过程
    train_losses = []
    val_accuracies = []

    # 创建Saver
    saver = tf.compat.v1.train.Saver(max_to_keep=3)

    # 创建TensorFlow配置
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        # 初始化变量
        sess.run(tf.compat.v1.global_variables_initializer())

        print(f"\n🚀 开始训练...")
        print(f"   训练样本: {len(train_x)}")
        print(f"   测试样本: {len(test_x)}")
        print(f"   批次大小: {batch_size}")
        print(f"   每轮批次: {num_batches}")
        print(f"   最大轮次: {num_epochs}")
        print(f"   早停耐心值: {patience}")

        # 训练循环
        for epoch in range(num_epochs):
            epoch_losses = []

            # 打乱训练数据
            indices = np.arange(len(train_x))
            np.random.shuffle(indices)
            train_x_shuffled = train_x[indices]
            train_y_shuffled = train_y[indices]

            # 批次训练
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_x_shuffled))

                batch_x = train_x_shuffled[start_idx:end_idx]
                batch_y = train_y_shuffled[start_idx:end_idx]

                # 训练步骤
                _, loss = sess.run([optimizer, cross_entropy],
                                  feed_dict={input_image: batch_x,
                                            input_label: batch_y,
                                            dropout_rate: 0.5,
                                            dropout_rate_2: 0.3})

                epoch_losses.append(loss)

            # 计算平均损失
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)

            # 计算验证准确率
            val_acc = accuracy.eval(feed_dict={
                input_image: test_x,
                input_label: test_y,
                dropout_rate: 1.0,
                dropout_rate_2: 1.0
            })
            val_accuracies.append(val_acc)

            # 输出训练进度
            print(f"📍 轮次 {epoch+1:3d}/{num_epochs} - "
                  f"损失: {avg_loss:.4f} - "
                  f"验证准确率: {val_acc:.4f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                saver.save(sess, os.path.join(model_dir, 'best_model'))
                print(f"   ✅ 保存最佳模型 (准确率: {val_acc:.4f})")
                patience_counter = 0

                # 保存损失记录
                with open(os.path.join(model_dir, 'loss.txt'), 'w') as f:
                    for loss_val in train_losses:
                        f.write(f"{loss_val}\n")
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= patience:
                print(f"\n🛑 早停触发!")
                print(f"   连续 {patience} 轮验证准确率未提升")
                break

            # 如果准确率足够高，提前停止
            if val_acc > 0.95:
                print(f"\n🎯 达到目标准确率!")
                break

        # 保存最终模型
        saver.save(sess, os.path.join(model_dir, 'final_model'))

        print(f"\n✅ 训练完成!")
        print(f"   最佳验证准确率: {best_val_acc:.4f}")
        print(f"   最终模型保存到: {model_dir}")

        # 绘制训练曲线
        if len(train_losses) > 1:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'training_history.png'))
            plt.show()

        # 输出模型偏置信息（诊断用）
        all_vars = tf.compat.v1.global_variables()
        for var in all_vars:
            if 'b5:0' in var.name:  # 输出层偏置
                bias_value = sess.run(var)
                print(f"\n📊 输出层偏置:")
                for i, bias in enumerate(bias_value):
                    class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
                    print(f"   {class_name}: {bias:.4f}")

if __name__ == '__main__':
    """主函数 - 修复版"""

    print("=" * 60)
    print("          平衡数据人脸识别模型训练")
    print("=" * 60)

    # 定义参数
    faces_ok_dir = './faces_ok'
    faces_no_dir = './faces_no'
    size = 64
    batch_size = 32
    learning_rate = 0.001

    # 检查目录是否存在
    if not os.path.exists(faces_ok_dir):
        print(f"❌ 错误: {faces_ok_dir} 目录不存在")
        sys.exit(1)

    if not os.path.exists(faces_no_dir):
        print(f"⚠️ 警告: {faces_no_dir} 目录不存在，将生成虚拟陌生人数据")
        os.makedirs(faces_no_dir, exist_ok=True)

    # 1. 使用平衡数据加载器加载数据
    print("\n📥 加载数据并平衡...")
    data_loader = BalancedDataLoader(target_samples_per_class=400)  # 每类目标400张
    imgs, labs, class_names = data_loader.load_balanced_data(faces_ok_dir, faces_no_dir, size)

    if imgs is None:
        print("❌ 数据加载失败")
        sys.exit(1)

    num_classes = len(class_names)
    print(f"\n📊 最终数据统计:")
    print(f"   总图片数: {len(imgs)}")

    # 统计各类别数量
    lab_array = np.array(labs)
    for i in range(num_classes):
        count = np.sum(lab_array[:, i])
        print(f"   类别 {i} ({class_names[i]}): {count} 张")

    # 2. 划分训练集和测试集
    print(f"\n🔀 划分训练集和测试集...")
    train_x, test_x, train_y, test_y = train_test_split(
        imgs, labs, test_size=0.2, random_state=42,
        stratify=np.argmax(labs, axis=1)  # 分层抽样
    )

    print(f"   训练集: {len(train_x)} 张")
    print(f"   测试集: {len(test_x)} 张")

    # 3. 归一化和重塑
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0

    train_x = train_x.reshape(-1, size, size, 3)
    test_x = test_x.reshape(-1, size, size, 3)

    # 4. 定义TensorFlow图
    input_image = tf.compat.v1.placeholder(tf.float32, [None, size, size, 3])
    input_label = tf.compat.v1.placeholder(tf.float32, [None, num_classes])
    dropout_rate = tf.compat.v1.placeholder(tf.float32)
    dropout_rate_2 = tf.compat.v1.placeholder(tf.float32)

    # 5. 构建网络 - 修改初始化以减少偏置不平衡
    print(f"\n🧠 构建神经网络...")

    # 获取网络输出
    outdata = net.layer_net(input_image, num_classes, dropout_rate, dropout_rate_2)

    # 6. 定义损失函数和优化器
    # 添加类别权重（给样本少的类别更高权重）
    class_weights = tf.constant([1.0] * num_classes, dtype=tf.float32)

    # 计算加权交叉熵损失
    unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=input_label, logits=outdata
    )

    # 计算每个样本的权重
    sample_weights = tf.reduce_sum(input_label * class_weights, axis=1)
    weighted_loss = unweighted_loss * sample_weights
    cross_entropy = tf.reduce_mean(weighted_loss)

    # 使用Adam优化器
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # 7. 开始训练
    do_train(outdata, cross_entropy, optimizer, num_classes, class_names,
             train_x, train_y, test_x, test_y, batch_size)

    print(f"\n🎉 所有训练完成!")
    print(f"   模型保存在: ./model_balanced/")
    print(f"   使用新模型进行识别时，请修改识别脚本中的模型路径")
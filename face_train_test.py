"""
model_reset_tool_fixed.py - 模型重置和修复工具（修复版）
解决TensorFlow兼容性问题
"""

import os
import shutil
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import random
import glob
import matplotlib.pyplot as plt

# 禁用TensorFlow 2.x行为，启用v1兼容模式
tf.disable_v2_behavior()
tf.disable_eager_execution()


def reset_and_retrain():
    """重置并重新训练模型"""
    print("=" * 60)
    print("    模型重置和修复工具（修复版）")
    print("=" * 60)

    # 删除旧模型
    model_dir = './model_fixed'
    if os.path.exists(model_dir):
        print(f"🗑️  删除旧模型目录: {model_dir}")
        shutil.rmtree(model_dir)

    # 创建新模型目录
    os.makedirs(model_dir, exist_ok=True)

    # 重新训练
    return train_fixed_model()


def verify_data():
    """验证数据正确性"""
    print("\n🔍 验证训练数据...")

    # 检查用户数据
    user_dir = './faces_user'
    if not os.path.exists(user_dir):
        print(f"❌ 用户数据目录不存在: {user_dir}")
        return False

    user_count = 0
    user_subdirs = []
    for item in os.listdir(user_dir):
        item_path = os.path.join(user_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            user_count += len(files)
            user_subdirs.append((item, len(files)))

    print(f"   用户数据: {user_count} 张图片")
    for subdir, count in user_subdirs:
        print(f"     - {subdir}: {count} 张")

    # 检查陌生人数据
    stranger_dir = './faces_strangers'
    if not os.path.exists(stranger_dir):
        print(f"⚠️ 陌生人数据目录不存在: {stranger_dir}")
        stranger_count = 0
    else:
        stranger_count = 0
        for item in os.listdir(stranger_dir):
            item_path = os.path.join(stranger_dir, item)
            if os.path.isdir(item_path):
                for img_file in os.listdir(item_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(item_path, img_file)
                        # 检查文件是否存在且可读
                        if os.path.exists(img_path) and os.path.isfile(img_path):
                            try:
                                img = cv2.imread(img_path)
                                if img is not None:
                                    stranger_count += 1
                            except:
                                print(f"   跳过无法读取的文件: {img_path}")
            elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(stranger_dir, item)
                if os.path.exists(img_path) and os.path.isfile(img_path):
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            stranger_count += 1
                    except:
                        print(f"   跳过无法读取的文件: {img_path}")

    print(f"   陌生人数据: {stranger_count} 张图片")

    if user_count == 0:
        print("❌ 没有找到用户数据")
        return False

    if stranger_count == 0:
        print("⚠️ 没有找到陌生人数据，模型将只能识别用户")

    return True


def load_and_prepare_data():
    """加载并准备数据"""
    print("\n📥 加载和准备数据...")

    # 加载用户数据
    user_imgs = []
    user_labels = []

    user_dir = './faces_user'
    for item in os.listdir(user_dir):
        item_path = os.path.join(user_dir, item)
        if os.path.isdir(item_path):
            for img_file in os.listdir(item_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(item_path, img_file)
                    if os.path.exists(img_path) and os.path.isfile(img_path):
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                # 预处理
                                img = cv2.resize(img, (64, 64))
                                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                                user_imgs.append(img)
                                user_labels.append([1.0, 0.0])  # 用户标签
                        except:
                            print(f"   跳过无法读取的用户图片: {img_path}")

    print(f"   加载用户图片: {len(user_imgs)} 张")

    # 加载陌生人数据
    stranger_imgs = []
    stranger_labels = []

    stranger_dir = './faces_strangers'
    if os.path.exists(stranger_dir):
        for item in os.listdir(stranger_dir):
            item_path = os.path.join(stranger_dir, item)
            if os.path.isdir(item_path):
                for img_file in os.listdir(item_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(item_path, img_file)
                        if os.path.exists(img_path) and os.path.isfile(img_path):
                            try:
                                img = cv2.imread(img_path)
                                if img is not None:
                                    # 预处理
                                    img = cv2.resize(img, (64, 64))
                                    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                                    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                                    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                                    stranger_imgs.append(img)
                                    stranger_labels.append([0.0, 1.0])  # 陌生人标签
                            except:
                                print(f"   跳过无法读取的陌生人图片: {img_path}")
            elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(stranger_dir, item)
                if os.path.exists(img_path) and os.path.isfile(img_path):
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            # 预处理
                            img = cv2.resize(img, (64, 64))
                            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                            stranger_imgs.append(img)
                            stranger_labels.append([0.0, 1.0])  # 陌生人标签
                    except:
                        print(f"   跳过无法读取的陌生人图片: {img_path}")

    print(f"   加载陌生人图片: {len(stranger_imgs)} 张")

    # 平衡数据 - 如果用户数据比陌生人多，对陌生人进行数据增强
    if len(stranger_imgs) < len(user_imgs) and len(stranger_imgs) > 0:
        print(f"   平衡数据: 陌生人数据较少，进行数据增强")
        ratio = len(user_imgs) // len(stranger_imgs)
        if ratio > 1:
            additional_stranger_imgs = []
            additional_stranger_labels = []

            for _ in range(ratio - 1):
                for img, label in zip(stranger_imgs, stranger_labels):
                    # 简单的数据增强
                    aug_img = cv2.flip(img, 1)  # 水平翻转
                    additional_stranger_imgs.append(aug_img)
                    additional_stranger_labels.append(label)

            stranger_imgs.extend(additional_stranger_imgs)
            stranger_labels.extend(additional_stranger_labels)

    # 合并数据
    all_imgs = user_imgs + stranger_imgs
    all_labels = user_labels + stranger_labels

    print(f"   总数据: {len(all_imgs)} 张")
    print(f"   用户标签: {sum(1 for label in all_labels if label[0] == 1.0)}")
    print(f"   陌生人标签: {sum(1 for label in all_labels if label[1] == 1.0)}")

    # 转换为numpy数组
    all_imgs = np.array(all_imgs, dtype=np.float32) / 255.0
    all_labels = np.array(all_labels, dtype=np.float32)

    # 重塑
    all_imgs = all_imgs.reshape(-1, 64, 64, 3)

    return all_imgs, all_labels


def train_fixed_model():
    """训练修复后的模型"""
    print("\n🚀 开始训练修复模型...")

    # 验证数据
    if not verify_data():
        return False

    # 加载数据
    all_imgs, all_labels = load_and_prepare_data()

    if len(all_imgs) == 0:
        print("❌ 没有加载到任何数据")
        return False

    # 划分训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(
        all_imgs, all_labels, test_size=0.2, random_state=42,
        stratify=np.argmax(all_labels, axis=1)
    )

    print(f"   训练集: {len(train_x)} 张")
    print(f"   测试集: {len(test_x)} 张")

    # 构建模型
    print("\n🧠 构建模型...")

    # 重置计算图
    tf.reset_default_graph()

    # 定义占位符
    input_image = tf.placeholder(tf.float32, [None, 64, 64, 3])
    input_label = tf.placeholder(tf.float32, [None, 2])  # 2个类别
    dropout_rate = tf.placeholder(tf.float32)
    dropout_rate_2 = tf.placeholder(tf.float32)

    # 构建网络（使用简单的CNN结构）
    conv1 = tf.layers.conv2d(input_image, 32, 3, activation=tf.nn.relu, padding='same')
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, padding='same')
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

    conv3 = tf.layers.conv2d(pool2, 128, 3, activation=tf.nn.relu, padding='same')
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2)

    flat = tf.layers.flatten(pool3)

    dense1 = tf.layers.dense(flat, 512, activation=tf.nn.relu)
    dropout1 = tf.nn.dropout(dense1, rate=1-dropout_rate)

    dense2 = tf.layers.dense(dropout1, 256, activation=tf.nn.relu)
    dropout2 = tf.nn.dropout(dense2, rate=1-dropout_rate_2)

    outdata = tf.layers.dense(dropout2, 2)  # 2个输出

    # 损失函数
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=outdata,
            labels=input_label
        )
    )

    # 优化器
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 准确率
    correct_prediction = tf.equal(tf.argmax(outdata, 1), tf.argmax(input_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Saver
    saver = tf.train.Saver()

    # 训练
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("\n📈 开始训练...")
        best_val_acc = 0
        patience_counter = 0
        patience = 10

        batch_size = 32
        num_epochs = 100

        for epoch in range(num_epochs):
            # 打乱训练数据
            indices = np.arange(len(train_x))
            np.random.shuffle(indices)
            train_x_shuffled = train_x[indices]
            train_y_shuffled = train_y[indices]

            # 批次训练
            total_loss = 0
            num_batches = len(train_x_shuffled) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(train_x_shuffled))

                batch_x = train_x_shuffled[start_idx:end_idx]
                batch_y = train_y_shuffled[start_idx:end_idx]

                _, loss = sess.run([optimizer, cross_entropy],
                                  feed_dict={
                                      input_image: batch_x,
                                      input_label: batch_y,
                                      dropout_rate: 0.5,
                                      dropout_rate_2: 0.3
                                  })

                total_loss += loss

            avg_loss = total_loss / num_batches

            # 验证准确率
            val_acc = accuracy.eval(feed_dict={
                input_image: test_x,
                input_label: test_y,
                dropout_rate: 1.0,
                dropout_rate_2: 1.0
            })

            print(f"   轮次 {epoch+1:3d}/{num_epochs} - 损失: {avg_loss:.4f} - 验证准确率: {val_acc:.4f}")

            # 检查异常值
            if np.isnan(avg_loss) or np.isinf(avg_loss):
                print("   ❌ 检测到异常损失值，停止训练")
                break

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                saver.save(sess, './model_fixed/best_model')
                patience_counter = 0
                print(f"   ✅ 保存最佳模型 (准确率: {val_acc:.4f})")
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= patience:
                print(f"   🛑 早停触发，连续 {patience} 轮未提升")
                break

        # 保存最终模型
        saver.save(sess, './model_fixed/final_model')

        # 保存类别名称
        with open('./model_fixed/class_names.txt', 'w', encoding='utf-8') as f:
            f.write('用户\n')
            f.write('陌生人\n')

        print(f"\n✅ 训练完成!")
        print(f"   最佳验证准确率: {best_val_acc:.4f}")
        print(f"   模型保存到: ./model_fixed/")

        return True


def test_model():
    """测试修复后的模型"""
    print("\n🔍 测试修复后的模型...")

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.disable_eager_execution()

    import numpy as np
    import cv2

    # 加载模型
    try:
        # 重置图
        tf.reset_default_graph()

        # 定义网络结构（与训练时相同）
        input_image = tf.placeholder(tf.float32, [None, 64, 64, 3])
        dropout_rate = tf.placeholder(tf.float32)
        dropout_rate_2 = tf.placeholder(tf.float32)

        # 重建网络
        conv1 = tf.layers.conv2d(input_image, 32, 3, activation=tf.nn.relu, padding='same')
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, padding='same')
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

        conv3 = tf.layers.conv2d(pool2, 128, 3, activation=tf.nn.relu, padding='same')
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2)

        flat = tf.layers.flatten(pool3)

        dense1 = tf.layers.dense(flat, 512, activation=tf.nn.relu)
        dropout1 = tf.nn.dropout(dense1, rate=1-dropout_rate)

        dense2 = tf.layers.dense(dropout1, 256, activation=tf.nn.relu)
        dropout2 = tf.nn.dropout(dense2, rate=1-dropout_rate_2)

        outdata = tf.layers.dense(dropout2, 2)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # 恢复模型
            saver.restore(sess, './model_fixed/best_model')
            print("   ✅ 模型加载成功")

            # 测试一些用户图片
            user_dir = './faces_user'
            for item in os.listdir(user_dir):
                item_path = os.path.join(user_dir, item)
                if os.path.isdir(item_path):
                    for img_file in os.listdir(item_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(item_path, img_file)
                            if os.path.exists(img_path) and os.path.isfile(img_path):
                                try:
                                    img = cv2.imread(img_path)
                                    if img is not None:
                                        # 预处理
                                        img = cv2.resize(img, (64, 64))
                                        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                                        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                                        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                                        img_normalized = img.astype(np.float32) / 255.0
                                        img_batch = np.expand_dims(img_normalized, axis=0)

                                        # 预测
                                        logits = sess.run(outdata,
                                                        feed_dict={
                                                            input_image: img_batch,
                                                            dropout_rate: 1.0,
                                                            dropout_rate_2: 1.0
                                                        })

                                        # 计算概率
                                        exp_logits = np.exp(logits - np.max(logits))
                                        probs = exp_logits / np.sum(exp_logits)

                                        print(f"   测试图片: {img_file}")
                                        print(f"   原始logits: {logits[0]}")
                                        print(f"   概率分布: 用户={probs[0][0]:.4f}, 陌生人={probs[0][1]:.4f}")

                                        predicted_class = np.argmax(probs[0])
                                        class_names = ['用户', '陌生人']
                                        print(f"   预测结果: {class_names[predicted_class]}")
                                        print("   " + "-" * 40)

                                        # 只测试一张用户图片
                                        break
                                except:
                                    print(f"   跳过无法处理的图片: {img_path}")
                    break

            return True

    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 重置并训练模型
    success = reset_and_retrain()

    if success:
        print("\n🎉 模型重置训练成功!")

        # 测试模型
        test_model()

        print("\n✅ 修复完成！请使用新的模型 ./model_fixed/ 进行人脸识别")
    else:
        print("\n❌ 模型重置失败")




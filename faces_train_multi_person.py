"""--------------------------------------------------------------
二、CNN模型训练 - 多类别人脸识别版本
训练模型：共八层神经网络，卷积层特征提取，池化层降维,全连接层进行分类。
训练数据：根据实际数据量调整
多类别：每个faces_ok子目录为一个类别 + 其他人脸类别
学习率：0.001
损失函数：交叉熵
优化器：Adam
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

"""数据增强函数"""
def data_augmentation(image):
    """数据增强：随机翻转、亮度调整"""
    img = image.copy()

    # 随机水平翻转
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # 随机亮度调整
    brightness_factor = random.uniform(0.8, 1.2)
    img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)

    return img

"""定义读取人脸数据函数，根据不同的人名，分配不同的onehot值"""
def get_images_labels(in_path, height, width, label_idx, num_classes, augment=True):
    for file in os.listdir(in_path):
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            # 使用os.path.join来处理路径，兼容不同操作系统
            file_path = os.path.join(in_path, file)
            img = cv2.imread(file_path)
            if img is not None:  # 确保图片加载成功
                print(f"成功加载图片: {file_path}")
                t, b, l, r = units.img_padding(img)
                """放大图片扩充图片边缘部分"""
                img_big = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                img_big = cv2.resize(img_big, (height, width))

                # 添加原始图片
                imgs.append(img_big)

                # 创建one-hot标签
                label_onehot = [0] * num_classes
                label_onehot[label_idx] = 1
                labs.append(label_onehot)

                # 如果启用增强，添加增强后的图片
                if augment:
                    augmented_img = data_augmentation(img_big.copy())
                    imgs.append(augmented_img)

                    # 同样的标签
                    label_onehot = [0] * num_classes
                    label_onehot[label_idx] = 1
                    labs.append(label_onehot)
            else:
                print(f"警告：无法加载图片: {file_path}")

"""定义训练函数"""
def do_train(outdata, cross_entropy, optimizer, num_classes, class_names):
    """求得准确率，比较标签是否相等，再求的所有数的平均值"""
    correct_prediction = tf.equal(tf.argmax(outdata, 1), tf.argmax(input_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy : ", accuracy)

    steps = []
    losss = []
    test_accuracies = []

    # 确保模型保存目录存在
    model_dir = './model_multi_class'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 保存类别名称
    with open(os.path.join(model_dir, 'class_names.txt'), 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(name + '\n')

    saver = tf.compat.v1.train.Saver()

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        best_test_acc = 0
        patience = 5  # 早停耐心值
        patience_counter = 0

        for n in range(50):  # 限制总轮次
            for i in range(num_batch):
                batch_x = train_x_normalization[i*batch_size: (i+1)*batch_size]          # 图片
                batch_y = train_y[i*batch_size: (i+1)*batch_size]          # 标签
                _, loss = sess.run([optimizer, cross_entropy],
                                   feed_dict={input_image: batch_x, input_label: batch_y,
                                              dropout_rate: 0.6, dropout_rate_2: 0.4})  # 使用dropout
                steps.append(n * num_batch + i)
                losss.append(float(loss))
                print("step:%d,  loss:%g" % (n * num_batch + i, loss))

                # 确保loss.txt文件的目录存在
                loss_file_dir = os.path.dirname('./model_multi_class/loss.txt')
                if not os.path.exists(loss_file_dir):
                    os.makedirs(loss_file_dir)

                # 写入最新的loss值
                with open('./model_multi_class/loss.txt', 'w') as f:
                    for l in losss:
                        f.write(str(l) + '\n')

                if (n*num_batch+i) % 100 == 0:
                    # 测试准确率（不使用dropout）
                    acc = accuracy.eval({input_image: test_x_normalization, input_label: test_y,
                                        dropout_rate: 1.0, dropout_rate_2: 1.0})
                    test_accuracies.append(acc)
                    print("step:%d,  test_acc:%g" % (n*num_batch+i, acc))

                    # 保存最佳模型
                    if acc > best_test_acc and acc < 0.95:  # 限制最高准确率防止过拟合
                        best_test_acc = acc
                        saver.save(sess, './model_multi_class/best_model',
                                  global_step=n*num_batch+i)
                        print(f"新的最佳测试准确率: {acc:.4f}")

                    # 早停机制
                    if len(test_accuracies) > 1:
                        if acc <= test_accuracies[-2]:  # 如果准确率没有提升
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"早停触发：连续{patience}次测试准确率未提升，停止训练")
                                return  # 提前结束训练
                        else:
                            patience_counter = 0

                    # 如果测试准确率过高（>90%），停止训练以防止过拟合
                    if acc > 0.90:
                        print(f"测试准确率过高({acc:.4f})，停止训练以防止过拟合")
                        return

            # 每轮结束后评估
            avg_loss = np.mean(losss[-num_batch:]) if len(losss) >= num_batch else np.mean(losss)
            avg_acc = accuracy.eval({input_image: test_x_normalization, input_label: test_y,
                                    dropout_rate: 1.0, dropout_rate_2: 1.0})
            print(f"第{n}轮 - 平均训练损失: {avg_loss:.4f}, 测试准确率: {avg_acc:.4f}")

            # 如果准确率过高，提前停止
            if avg_acc > 0.88:
                print(f"平均测试准确率过高({avg_acc:.4f})，停止训练以防止过拟合")
                break


if __name__ == '__main__':
    """定义参数"""
    # 自动获取faces_ok目录下的所有子目录作为多个人脸类别
    faces_ok_dir = './faces_ok'
    if not os.path.exists(faces_ok_dir):
        print(f"错误：目录 {faces_ok_dir} 不存在，请先创建该目录并放入子目录")
        print("请在 {faces_ok_dir} 目录下创建包含人脸图片的子目录")
        sys.exit(1)

    # 获取faces_ok目录下的所有子目录
    multi_face_paths = []
    for item in os.listdir(faces_ok_dir):
        item_path = os.path.join(faces_ok_dir, item)
        if os.path.isdir(item_path):
            multi_face_paths.append(item_path)

    if len(multi_face_paths) == 0:
        print(f"错误：在 {faces_ok_dir} 目录下未找到任何子目录")
        print(f"请在 {faces_ok_dir} 目录下创建包含人脸图片的子目录")
        sys.exit(1)

    print(f"找到 {len(multi_face_paths)} 个多人脸类别目录：")
    for i, path in enumerate(multi_face_paths):
        print(f"  {i+1}. {path}")

    # 其他人脸目录（作为单独的一个类别）
    faces_no_path = './faces_no'     # 作为类别 num_classes-1
    num_classes = len(multi_face_paths) + 1  # 每个子目录一个类别 + 其他人脸类别
    class_names = []  # 存储类别名称

    # 添加faces_ok中每个子目录的名称
    for path in multi_face_paths:
        class_names.append(os.path.basename(path))  # 使用目录名作为类别名

    # 添加"陌生人"作为最后一个类别
    class_names.append("陌生人")

    print(f"总类别数: {num_classes}")
    print(f"类别名称: {class_names}")

    batch_size = 32  # 减小批次大小
    learning_rate = 0.0005  # 降低学习率

    size = 64  # 图片大小64*64*3
    imgs = []  # 存放人脸图片
    labs = []  # 存放人脸图片对应的标签

    """1、读取人脸数据、分配标签"""
    # 检查多人目录是否存在（已经在上面获取了）
    for path in multi_face_paths:
        if not os.path.exists(path):
            print(f"错误：目录 {path} 不存在")
            sys.exit(1)

    # 检查其他人脸目录是否存在
    if not os.path.exists(faces_no_path):
        print(f"错误：目录 {faces_no_path} 不存在，请先准备数据")
        print("请创建目录并放入其他人的脸图片")
        sys.exit(1)

    # 添加多人脸数据（每个子目录为一个类别）
    for idx, path in enumerate(multi_face_paths):
        initial_count = len(imgs)
        get_images_labels(path, size, size, idx, num_classes, augment=True)
        count_added = len(imgs) - initial_count
        print(f"从 {os.path.basename(path)} 添加了 {count_added} 张图片（含增强）")

    # 添加其他人脸数据（作为最后一个类别）
    initial_count = len(imgs)
    get_images_labels(faces_no_path, size, size, num_classes-1, num_classes, augment=False)
    count_added = len(imgs) - initial_count
    print(f"从 {os.path.basename(faces_no_path)} 添加了 {count_added} 张图片")

    imgs = np.array(imgs)                   # 将图片数据与标签转换成数组
    print("总图片数量: ", len(imgs))

    # 统计各类别数量
    lab_array = np.array(labs)
    for i in range(num_classes):
        count = np.sum(lab_array[:, i])
        print(f"类别 {i} ({class_names[i]}): {count} 张图片")

    labs = np.array(labs)

    print("labs shape: ", labs.shape)
    print("len labs: ", len(labs))

    """2、随机划分测试集与训练集,按照1：8的比列"""
    train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=1 / 8,
                                                        random_state=random.randint(0, 100))
    print("[train_x, test_x, train_y, test_y]: ", [len(train_x), len(test_x), len(train_y), len(test_y)])
    train_x_reshape = train_x.reshape(train_x.shape[0], size, size, 3)  # 参数：图片数据的总数，图片的高、宽、通道
    test_x_reshape = test_x.reshape(test_x.shape[0], size, size, 3)

    """3、归一化"""
    train_x_normalization = train_x_reshape.astype('float32') / 255.0
    test_x_normalization = test_x_reshape.astype('float32') / 255.0
    print('len(train_x_normalization): ', len(train_x_normalization))
    print('len(test_x_normalization): ', len(test_x_normalization))

    # 定义训练参数
    num_batch = len(train_x_normalization) // batch_size
    print('num_batch: ', num_batch)
    input_image = tf.compat.v1.placeholder(tf.float32, [None, size, size, 3])  # 输入X：64*64*3
    input_label = tf.compat.v1.placeholder(tf.float32, [None, num_classes])  # 输出Y_：1*num_classes
    dropout_rate = tf.compat.v1.placeholder(tf.float32)  # 定义
    dropout_rate_2 = tf.compat.v1.placeholder(tf.float32)  # 定义

    """神经网络输出"""
    outdata = net.layer_net(input_image, num_classes, dropout_rate, dropout_rate_2)
    """定义损失函数为交叉熵"""
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_label, logits=outdata))
    """采用Adam优化器"""
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    """4、进行训练"""
    do_train(outdata, cross_entropy, optimizer, num_classes, class_names)




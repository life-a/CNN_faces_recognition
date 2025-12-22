"""---------------------------------------------------------
四、人脸识别 - 多类别人脸识别版本（修复中文乱码）
1、打开摄像头，获取图片并灰度化
2、人脸检测
3、加载卷积神经网络模型（多类别识别模型）
4、多类别人脸识别（显示具体子目录名称）
------------------------------------------------------------"""
import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
import net
import units

# 禁用eager execution以支持占位符
tf.compat.v1.disable_eager_execution()

"""定义人脸识别函数"""
def face_recognise(image, sess, outdata, input_image, dropout_rate, dropout_rate_2):
    # 获取原始输出值
    raw_output = sess.run(outdata, feed_dict={input_image: [image/255.0], dropout_rate: 1.0, dropout_rate_2: 1.0})

    # 获取softmax概率
    softmax_probs = sess.run(tf.nn.softmax(outdata), feed_dict={input_image: [image/255.0], dropout_rate: 1.0, dropout_rate_2: 1.0})

    # 获取预测类别
    predicted_class = sess.run(tf.argmax(outdata, 1), feed_dict={input_image: [image/255.0], dropout_rate: 1.0, dropout_rate_2: 1.0})

    # 获取置信度
    confidence = np.max(softmax_probs)

    return predicted_class[0], confidence, softmax_probs[0]

def main():
    """主函数：多类别人脸识别"""
    # 创建会话并加载模型
    sess = tf.compat.v1.Session()

    # 检查多类别识别模型是否存在
    model_path = './model_multi_class/'
    checkpoint_path = tf.compat.v1.train.latest_checkpoint(model_path)

    if checkpoint_path is None:
        print(f"错误：未找到多类别识别模型，请先运行多类别训练脚本")
        print("请确保已使用多类别训练脚本完成训练")
        print("需要的目录结构：")
        print("- ./faces_ok/ (包含多个子目录，每个子目录包含一个人的图片)")
        print("- ./faces_no/ (包含其他人的图片)")
        sys.exit(1)

    # 读取类别名称
    class_names_file = os.path.join(model_path, 'class_names.txt')
    if not os.path.exists(class_names_file):
        print(f"错误：找不到类别名称文件 {class_names_file}")
        sys.exit(1)

    with open(class_names_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    num_classes = len(class_names)
    print(f"加载了 {num_classes} 个类别: {class_names}")

    # 定义参数
    size = 64      # 图片大小64*64*3

    # 使用compat.v1定义占位符
    input_image = tf.compat.v1.placeholder(tf.float32, [None, size, size, 3])
    dropout_rate = tf.compat.v1.placeholder(tf.float32)
    dropout_rate_2 = tf.compat.v1.placeholder(tf.float32)

    # 构建网络
    outdata = net.layer_net(input_image, num_classes, dropout_rate, dropout_rate_2)

    # 创建会话并加载模型
    saver = tf.compat.v1.train.Saver()

    print(f"正在加载多类别识别模型: {checkpoint_path}")
    saver.restore(sess, checkpoint_path)
    print("多类别识别模型加载成功")

    # 初始化人脸检测器
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误：无法打开摄像头")
        sys.exit(1)

    print("多类别人脸识别系统启动，按ESC退出...")
    print("系统将显示具体的子目录名称，如 xsx, hcs 等")
    print("陌生人类别将显示为 '陌生人'")

    while True:
        ret, img = cap.read()
        if not ret:
            print("错误：无法从摄像头读取图像")
            break

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray_image, 1)

        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            # 边界检查
            if x1 >= img.shape[0] or y1 >= img.shape[0] or x2 >= img.shape[1] or y2 >= img.shape[1]:
                continue

            # 提取并调整人脸区域
            face = img[x1:y1, x2:y2]
            face_resized = cv2.resize(face, (size, size))

            # 识别人脸
            predicted_class_idx, confidence, all_probs = face_recognise(
                face_resized, sess, outdata, input_image, dropout_rate, dropout_rate_2
            )

            # 获取类别名称
            if predicted_class_idx < len(class_names):
                person_name = class_names[predicted_class_idx]
            else:
                person_name = "未知"

            # 使用英文标签避免控制台乱码问题
            print_result = f'Recognition Result: {person_name}, Confidence: {confidence:.3f}'
            print(print_result)

            # 显示各类别概率
            prob_details = []
            for i in range(len(all_probs)):
                class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
                prob_details.append(f"{class_name}={all_probs[i]:.3f}")
            prob_str = ", ".join(prob_details)
            print(f'Class Probabilities: {prob_str}')

            # 根据置信度设置颜色和标签
            if confidence > 0.55:  # 调整置信度阈值
                if person_name == "陌生人":
                    color = (0, 0, 255)      # 红色 - 陌生人
                    label = f"Stranger ({confidence:.2f})"
                else:
                    color = (0, 255, 0)      # 绿色 - 已知人员
                    label = f"User:{person_name} ({confidence:.2f})"
            else:
                color = (128, 128, 128)      # 灰色 - 置信度低
                label = "Low Confidence"

            # 绘制人脸框和标签
            cv2.rectangle(img, (x2, x1), (y2, y1), color, 2)
            cv2.putText(img, label, (x2, y1+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Multi-Class Face Recognition', img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键退出
            break

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    sess.close()
    print("多类别人脸识别系统已关闭")

if __name__ == '__main__':
    main()

"""
使用说明：

1. 首先运行多类别人脸识别训练脚本训练模型
2. 确保数据目录结构正确：
   - ./faces_ok/xsx/ (xsx的图片)
   - ./faces_ok/hcs/ (hcs的图片)
   - ./faces_ok/其他子目录/ (其他人员的图片)
   - ./faces_no/ (其他人的图片)
3. 运行此识别脚本进行多类别实时识别

识别结果将显示具体的子目录名称，如"User:xsx"或"Stranger"。
"""




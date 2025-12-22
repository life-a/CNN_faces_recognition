"""---------------------------------------------------------
四、人脸识别 - 多类别人脸识别版本（修复版 - 已修正占位符错误）
修复问题：
1. 'Placeholder_1' Graph execution error
2. 总是识别为同一个人的问题
3. 添加时间平滑和历史投票
4. 动态置信度阈值
5. 更好的调试信息
------------------------------------------------------------"""
import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import sys
import time
from collections import deque, Counter

# 禁用eager execution以支持占位符
tf.compat.v1.disable_eager_execution()

# 导入net模块
import net

class ImprovedFaceRecognizer:
    """改进的人脸识别器，解决总是识别为同一个人的问题"""

    def __init__(self, model_path='./model_balanced/'):
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

        # 统计信息
        self.stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'predictions': [],
            'start_time': time.time()
        }

    def load_model(self):
        """加载训练好的平衡模型"""
        print(f"🔍 正在加载模型...")
        print(f"   模型路径: {self.model_path}")

        # 检查模型目录是否存在
        if not os.path.exists(self.model_path):
            print(f"❌ 错误: 模型目录 {self.model_path} 不存在")
            print(f"   请先运行训练脚本 face_train_multi_person.py")
            return False

        # 读取类别名称
        class_names_file = os.path.join(self.model_path, 'class_names.txt')
        if not os.path.exists(class_names_file):
            print(f"❌ 错误: 找不到类别名称文件 {class_names_file}")
            return False

        with open(class_names_file, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines() if line.strip()]

        self.num_classes = len(self.class_names)
        print(f"✅ 加载了 {self.num_classes} 个类别: {self.class_names}")

        # 为每个类别初始化动态阈值
        for i, name in enumerate(self.class_names):
            if name == "陌生人":
                self.class_thresholds[i] = 0.55  # 陌生人阈值较低
            else:
                self.class_thresholds[i] = 0.65  # 已知人员阈值较高

        # 🚨 修复关键：正确定义所有占位符
        size = 64
        self.input_image = tf.compat.v1.placeholder(tf.float32, [None, size, size, 3], name='input_image')
        self.dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
        self.dropout_rate_2 = tf.compat.v1.placeholder(tf.float32, name='dropout_rate_2')

        # 构建网络
        self.outdata = net.layer_net(self.input_image, self.num_classes,
                                     self.dropout_rate, self.dropout_rate_2)

        # 创建会话
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        saver = tf.compat.v1.train.Saver()

        # 尝试加载最佳模型，如果不存在则加载最终模型
        model_to_load = None
        best_model_path = os.path.join(self.model_path, 'best_model')
        final_model_path = os.path.join(self.model_path, 'final_model')

        if os.path.exists(best_model_path + '.index'):
            model_to_load = best_model_path
            print(f"   加载最佳模型: {model_to_load}")
        elif os.path.exists(final_model_path + '.index'):
            model_to_load = final_model_path
            print(f"   加载最终模型: {model_to_load}")
        else:
            # 尝试查找任何checkpoint
            checkpoint = tf.train.latest_checkpoint(self.model_path)
            if checkpoint:
                model_to_load = checkpoint
                print(f"   加载最新模型: {model_to_load}")
            else:
                print(f"❌ 错误: 在 {self.model_path} 中找不到模型文件")
                return False

        try:
            saver.restore(self.sess, model_to_load)
            print(f"✅ 模型加载成功")

            # 测试模型是否正常工作
            test_input = np.random.randn(1, 64, 64, 3) * 0.1
            probs = self.sess.run(tf.nn.softmax(self.outdata),
                                 feed_dict={
                                     self.input_image: test_input,
                                     self.dropout_rate: 1.0,  # 🚨 修复：使用正确的占位符变量
                                     self.dropout_rate_2: 1.0  # 🚨 修复：使用正确的占位符变量
                                 })
            print(f"   模型测试通过，输出shape: {probs.shape}")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

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
            # 🚨 修复关键：正确传递所有占位符
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
            print(f"⚠️ 模型推理错误: {e}")
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

    def print_stats(self):
        """打印统计信息"""
        total_time = time.time() - self.stats['start_time']

        print(f"\n📈 识别统计:")
        print(f"   总帧数: {self.stats['total_frames']}")
        print(f"   检测到人脸: {self.stats['faces_detected']} 次")
        print(f"   运行时间: {total_time:.2f}秒")

        if self.stats['total_frames'] > 0:
            fps = self.stats['total_frames'] / total_time
            print(f"   平均帧率: {fps:.2f} FPS")

        if self.stats['predictions']:
            pred_counter = Counter(self.stats['predictions'])
            print(f"\n📊 预测分布:")
            for i in range(self.num_classes):
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                count = pred_counter.get(i, 0)
                percentage = count / len(self.stats['predictions']) * 100
                print(f"   {class_name}: {count}次 ({percentage:.1f}%)")

def main():
    """主函数"""

    print("=" * 60)
    print("          改进版多类别人脸识别系统 (已修复)")
    print("=" * 60)

    # 创建识别器实例 - 先尝试新的平衡模型
    recognizer = ImprovedFaceRecognizer(model_path='./model_balanced/')

    # 加载模型
    if not recognizer.load_model():
        print("尝试加载原始模型...")
        recognizer = ImprovedFaceRecognizer(model_path='./model_multi_class/')
        if not recognizer.load_model():
            print("❌ 无法加载任何模型，请先训练模型")
            print("运行命令: python face_train_multi_person.py")
            return

    # 初始化人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        recognizer.close()
        return

    print(f"\n🚀 人脸识别系统已启动")
    print("   按以下键操作:")
    print("   ESC  : 退出程序")
    print("   V    : 切换详细模式")
    print("   R    : 重置识别历史")
    print("   +    : 提高置信度阈值")
    print("   -    : 降低置信度阈值")
    print("   S    : 显示统计信息")
    print("   C    : 清除屏幕输出")
    print(f"\n📊 当前设置:")
    print(f"   基础阈值: {recognizer.base_threshold:.2f}")
    print(f"   类别数: {recognizer.num_classes}")

    # 控制变量
    show_details = False
    clear_console = False

    # 性能监控
    fps_counter = deque(maxlen=30)
    last_time = time.time()
    last_print_time = time.time()

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法从摄像头读取帧")
            break

        recognizer.stats['total_frames'] += 1

        # 计算FPS
        current_time = time.time()
        fps = 1.0 / (current_time - last_time) if current_time != last_time else 0
        fps_counter.append(fps)
        avg_fps = np.mean(fps_counter) if fps_counter else 0
        last_time = current_time

        # 镜像显示（更自然）
        frame = cv2.flip(frame, 1)

        # 转换为灰度图进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = detector(gray, 1)

        # 如果没有检测到人脸
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            recognizer.stats['faces_detected'] += len(faces)

        # 处理每个检测到的人脸
        for i, face in enumerate(faces):
            try:
                # 获取人脸边界框
                x1 = max(face.top(), 0)
                y1 = min(face.bottom(), frame.shape[0])
                x2 = max(face.left(), 0)
                y2 = min(face.right(), frame.shape[1])

                # 提取人脸区域
                face_img = frame[x1:y1, x2:y2]
                if face_img.size == 0:
                    continue

                # 识别人脸（使用时间平滑）
                predicted_class, confidence, all_probs = recognizer.recognize_with_smoothing(face_img)

                if predicted_class is None:
                    continue

                # 保存到统计
                recognizer.stats['predictions'].append(predicted_class)

                # 获取类别名称
                if predicted_class < len(recognizer.class_names):
                    person_name = recognizer.class_names[predicted_class]
                else:
                    person_name = "未知"

                # 获取显示颜色
                color = recognizer.get_class_color(predicted_class, confidence)

                # 准备显示文本
                if person_name == "陌生人":
                    display_text = f"Stranger ({confidence:.2f})"
                else:
                    display_text = f"{person_name} ({confidence:.2f})"

                # 绘制人脸边界框
                cv2.rectangle(frame, (x2, x1), (y2, y1), color, 2)

                # 绘制文本背景
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame,
                            (x2, x1 - text_size[1] - 10),
                            (x2 + text_size[0], x1),
                            color, -1)

                # 绘制文本
                cv2.putText(frame, display_text,
                          (x2, x1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          (255, 255, 255), 2)

                # 在右上角显示详细概率（如果开启详细模式）
                if show_details and i == 0 and all_probs is not None:  # 只显示第一个脸的详细概率
                    prob_text_y = 30

                    # 显示前3个概率
                    top_indices = np.argsort(all_probs)[-3:][::-1]
                    for idx in top_indices:
                        if idx < len(recognizer.class_names):
                            class_name = recognizer.class_names[idx]
                            prob = all_probs[idx]

                            prob_text = f"{class_name}: {prob:.3f}"
                            text_width = frame.shape[1] - 200

                            cv2.putText(frame, prob_text,
                                      (text_width, prob_text_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (255, 255, 255), 1)
                            prob_text_y += 20

                # 定期输出识别结果到控制台（每2秒一次）
                if current_time - last_print_time > 2.0 and i == 0:
                    if clear_console:
                        os.system('cls' if os.name == 'nt' else 'clear')
                        clear_console = False

                    print(f"[{time.strftime('%H:%M:%S')}] 识别: {person_name}, 置信度: {confidence:.4f}")
                    last_print_time = current_time

            except Exception as e:
                print(f"⚠️ 处理人脸时出错: {e}")
                continue

        # 显示状态信息
        info_y = 30

        # 显示FPS
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 30

        # 显示帧数
        frame_text = f"Frame: {recognizer.stats['total_frames']}"
        cv2.putText(frame, frame_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        info_y += 30

        # 显示检测到的人脸数
        faces_text = f"Faces: {len(faces)}"
        cv2.putText(frame, faces_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        info_y += 30

        # 显示历史长度
        history_text = f"History: {len(recognizer.prediction_history)}"
        cv2.putText(frame, history_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

        # 显示当前阈值
        threshold_text = f"Threshold: {recognizer.base_threshold:.2f}"
        cv2.putText(frame, threshold_text, (frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示模式状态
        if show_details:
            cv2.putText(frame, "DETAIL MODE", (frame.shape[1] - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 显示操作提示
        cv2.putText(frame, "ESC:quit, V:details, R:reset, +/-:threshold",
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 显示窗口
        window_title = "Fixed Face Recognition - Press ESC to quit"
        cv2.imshow(window_title, frame)

        # 处理按键输入
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC键
            break
        elif key == ord('v') or key == ord('V'):  # 切换详细模式
            show_details = not show_details
            print(f"详细模式: {'开启' if show_details else '关闭'}")
        elif key == ord('r') or key == ord('R'):  # 重置历史
            recognizer.prediction_history.clear()
            recognizer.confidence_history.clear()
            print("识别历史已重置")
        elif key == ord('+'):  # 提高阈值
            recognizer.base_threshold = min(0.9, recognizer.base_threshold + 0.05)
            for i in recognizer.class_thresholds:
                recognizer.class_thresholds[i] = min(0.9, recognizer.class_thresholds[i] + 0.05)
            print(f"置信度阈值提高到: {recognizer.base_threshold:.2f}")
        elif key == ord('-'):  # 降低阈值
            recognizer.base_threshold = max(0.3, recognizer.base_threshold - 0.05)
            for i in recognizer.class_thresholds:
                recognizer.class_thresholds[i] = max(0.3, recognizer.class_thresholds[i] - 0.05)
            print(f"置信度阈值降低到: {recognizer.base_threshold:.2f}")
        elif key == ord('s') or key == ord('S'):  # 显示统计
            recognizer.print_stats()
        elif key == ord('c') or key == ord('C'):  # 清除控制台
            clear_console = True
            os.system('cls' if os.name == 'nt' else 'clear')

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()

    # 打印最终统计
    print(f"\n{'='*60}")
    print("          识别系统已关闭")
    print(f"{'='*60}")
    recognizer.print_stats()
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
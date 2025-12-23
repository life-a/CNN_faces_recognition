"""-----------------------------------------
一、增强版人脸数据采集系统
核心功能：
1. 多人员采集：按人员名称保存到 faces_ok/人员名称/ 目录
2. 自动数据增强：每张原始图片自动生成3个增强版本
3. 进度显示：实时显示采集进度和状态
-----------------------------------------"""
import cv2
import dlib
import os
import random
import numpy as np
from datetime import datetime

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

    def capture_data(self, person_name, target_count=100, camera_index=0):
        """
        采集指定人员的人脸数据
        :param person_name: 人员名称（英文或拼音，不要用中文）
        :param target_count: 目标采集数量（原始+增强后的总数量）
        :param camera_index: 摄像头索引，0=默认，1=外接
        """
        # 创建保存目录：faces_ok/人员名称/
        save_dir = os.path.join('./faces_ok', person_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"开始采集 [{person_name}] 的人脸数据")
        print(f"目标数量: {target_count}张（含3倍增强）")
        print(f"保存目录: {save_dir}")
        print(f"{'='*60}")

        print("采集指南：")
        print("1. 正对摄像头，保持自然表情")
        print("2. 缓慢左右转动头部（增加角度多样性）")
        print("3. 可轻微抬头、低头")
        print("4. 在不同位置采集（避免单一背景）")
        print(f"\n操作控制：")
        print("  按 'S' 键：开始/暂停采集")
        print("  按 'Q' 键：结束采集")
        print(f"{'='*60}")

        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {camera_index}")
            return

        collecting = False
        saved_count = 0
        frame_skip = 3  # 每3帧采集一次，避免过于相似
        frame_counter = 0

        # 检查已有图片数量
        existing_files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))]
        if existing_files:
            saved_count = len(existing_files)
            print(f"📁 发现已有 {saved_count} 张图片，将继续追加采集")

        print(f"\n⏳ 准备就绪，按 'S' 键开始采集...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break

            frame_counter += 1
            display_frame = frame.copy()

            # 人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)

            # 显示采集状态
            status = "采集进行中" if collecting else "已暂停"
            status_color = (0, 255, 0) if collecting else (0, 0, 255)

            cv2.putText(display_frame, f"状态: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, f"人员: {person_name}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"已保存: {saved_count}/{target_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, "按 'S':开始/暂停  按 'Q':结束", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示人脸检测框
            face_detected = False
            for face in faces:
                x1 = max(face.top(), 0)
                y1 = min(face.bottom(), frame.shape[0])
                x2 = max(face.left(), 0)
                y2 = min(face.right(), frame.shape[1])

                cv2.rectangle(display_frame, (x2, x1), (y2, y1), (0, 255, 0), 2)
                face_detected = True

                # 采集逻辑
                if collecting and frame_counter % frame_skip == 0 and saved_count < target_count:
                    face_img = frame[x1:y1, x2:y2]
                    if face_img.size > 0:
                        # 调整到标准尺寸
                        face_resized = cv2.resize(face_img, (self.size, self.size))

                        # 生成时间戳作为文件名
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        base_filename = f"{person_name}_{timestamp}"

                        # 保存原始图片
                        original_path = os.path.join(save_dir, f"{base_filename}_orig.jpg")
                        cv2.imwrite(original_path, face_resized)
                        saved_count += 1

                        # 显示刚刚采集的图片
                        cv2.imshow('最新采集', face_resized)

                        # 自动生成3个增强版本
                        if saved_count < target_count:
                            augmentations = self.apply_augmentations(face_resized)
                            for i, aug_img in enumerate(augmentations):
                                if saved_count >= target_count:
                                    break

                                aug_path = os.path.join(save_dir, f"{base_filename}_aug{i+1}.jpg")
                                cv2.imwrite(aug_path, aug_img)
                                saved_count += 1

                        print(f"✅ 已保存 {saved_count}/{target_count}")

            if not face_detected and collecting:
                cv2.putText(display_frame, "未检测到人脸", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 绘制进度条
            if target_count > 0:
                progress = min(saved_count / target_count, 1.0)
                bar_width = 300
                bar_height = 20
                bar_x, bar_y = 10, 180

                # 背景条
                cv2.rectangle(display_frame, (bar_x, bar_y),
                             (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)

                # 进度填充（颜色根据进度变化）
                fill_width = int(bar_width * progress)
                if progress < 0.3:
                    fill_color = (0, 0, 255)    # 红色
                elif progress < 0.7:
                    fill_color = (0, 255, 255)  # 黄色
                else:
                    fill_color = (0, 255, 0)    # 绿色

                cv2.rectangle(display_frame, (bar_x, bar_y),
                             (bar_x + fill_width, bar_y + bar_height), fill_color, -1)

                # 进度百分比文本
                progress_text = f"进度: {progress*100:.1f}% ({saved_count}/{target_count})"
                text_x = bar_x + bar_width + 10
                text_y = bar_y + bar_height // 2 + 5
                cv2.putText(display_frame, progress_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示主窗口
            cv2.imshow(f'人脸采集 - {person_name} (S:开始/暂停, Q:结束)', display_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                collecting = not collecting
                print(f"{'▶️  开始采集' if collecting else '⏸️  暂停采集'}")
            elif key == ord('q') or key == ord('Q'):
                print(f"⏹️  结束 {person_name} 的数据采集")
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

        # 最终统计
        final_files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))]
        print(f"\n{'='*60}")
        print(f"✅ 采集完成！")
        print(f"   人员: {person_name}")
        print(f"   图片总数: {len(final_files)} 张")
        print(f"   保存目录: {save_dir}")
        print(f"{'='*60}")

        return len(final_files)

def main():
    """主程序"""
    print("=" * 60)
    print("       人脸数据采集系统（核心功能版）")
    print("=" * 60)
    print("功能说明：")
    print("  1. 为不同人员采集人脸数据")
    print("  2. 自动生成增强图片（每张原始图+3张增强图）")
    print("  3. 实时显示采集进度和状态")
    print("=" * 60)

    collector = FaceDataCollector(size=64)

    while True:
        print("\n" + "-" * 40)
        print("请选择操作：")
        print("  1. 开始新的人员数据采集")
        print("  2. 继续为已有人员追加采集")
        print("  3. 退出系统")

        choice = input("\n请输入选项 (1-3): ").strip()

        if choice == '1':
            print("\n【新人员数据采集】")
            print("-" * 30)

            person_name = input("请输入人员名称（英文或拼音）: ").strip()
            if not person_name:
                print("❌ 名称不能为空")
                continue

            # 检查是否已存在该人员目录
            person_dir = os.path.join('./faces_ok', person_name)
            if os.path.exists(person_dir):
                existing = len([f for f in os.listdir(person_dir)
                               if f.endswith(('.jpg', '.png'))])
                print(f"⚠️  已存在 {person_name} 的数据: {existing} 张")
                action = input("是否覆盖？(y=覆盖, n=改为追加): ").strip().lower()
                if action == 'y':
                    # 清空目录重新开始
                    for f in os.listdir(person_dir):
                        if f.endswith(('.jpg', '.png')):
                            os.remove(os.path.join(person_dir, f))
                else:
                    # 追加采集，直接进入下一步设置目标数量
                    pass

            # 设置采集目标
            target_input = input("请输入目标图片总数（建议100-500）: ").strip()
            try:
                target_count = int(target_input) if target_input else 100
                if target_count < 20:
                    print("⚠️  目标数量过少，建议至少20张")
                    target_count = 100
            except:
                target_count = 100
                print(f"⚠️  输入无效，使用默认值: {target_count}")

            print(f"\n📝 采集设置：")
            print(f"  人员: {person_name}")
            print(f"  目标: {target_count} 张图片")
            print(f"  说明: 每采集1张原始图，会自动生成3张增强图")
            print(f"       预计需要采集约 {max(1, target_count//4)} 次原始捕获")

            confirm = input("\n是否开始采集？(y/n): ").strip().lower()
            if confirm == 'y':
                collector.capture_data(person_name, target_count)

        elif choice == '2':
            print("\n【为已有人员追加采集】")
            print("-" * 30)

            # 查找已有的人员目录
            if not os.path.exists('./faces_ok'):
                print("❌ faces_ok 目录不存在，请先创建或选择选项1")
                continue

            person_dirs = []
            for item in os.listdir('./faces_ok'):
                item_path = os.path.join('./faces_ok', item)
                if os.path.isdir(item_path):
                    count = len([f for f in os.listdir(item_path)
                                if f.endswith(('.jpg', '.png'))])
                    person_dirs.append((item, count))

            if not person_dirs:
                print("❌ 未找到任何人员数据，请先选择选项1")
                continue

            print("已有人员列表：")
            for i, (name, count) in enumerate(person_dirs, 1):
                print(f"  {i}. {name}: {count} 张图片")

            try:
                selection = int(input(f"\n请选择人员 (1-{len(person_dirs)}): ").strip()) - 1
                if 0 <= selection < len(person_dirs):
                    person_name = person_dirs[selection][0]
                    current_count = person_dirs[selection][1]

                    print(f"\n当前 {person_name} 有 {current_count} 张图片")
                    add_input = input("希望再增加多少张？（总数将达）: ").strip()

                    try:
                        add_count = int(add_input) if add_input else 100
                        target_count = current_count + add_count

                        confirm = input(f"将为 {person_name} 追加采集至 {target_count} 张？(y/n): ").strip().lower()
                        if confirm == 'y':
                            collector.capture_data(person_name, target_count)
                    except:
                        print("❌ 输入无效")
                else:
                    print("❌ 选择无效")
            except:
                print("❌ 输入无效")

        elif choice == '3':
            print("\n退出系统，感谢使用！")
            break

        else:
            print("❌ 无效选项，请重新输入")

if __name__ == '__main__':
    main()
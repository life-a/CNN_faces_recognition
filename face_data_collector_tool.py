"""
face_data_collector_tool.py - 人脸数据采集工具类
重构数据采集功能，使其可复用
"""

import os
import cv2
import dlib
import random
from datetime import datetime
import shutil
import threading
from collections import deque
import numpy as np


class FaceDataCollectorTool:
    """人脸数据采集工具类"""
    
    def __init__(self, size=64):
        """
        初始化数据采集器
        
        Args:
            size: 人脸图像大小，默认为64x64
        """
        self.size = size
        self.detector = dlib.get_frontal_face_detector()
        
        # 采集状态
        self.is_collecting = False
        self.collection_count = 0
        self.target_count = 0
        self.current_user = ""
        self.save_dir = ""
        self.frame_skip = 1
        self.frame_counter = 0
        
        # 回调函数
        self.on_progress_update = None
        self.on_collection_complete = None
        self.on_info_update = None
    
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
    
    def prepare_collection(self, person_name, target_count=100):
        """
        准备数据采集
        
        Args:
            person_name: 人员名称（英文或拼音）
            target_count: 目标采集数量（原始+增强后的总数量）
            
        Returns:
            tuple: (target_count, save_dir, frame_skip, frame_counter)
        """
        # 创建保存目录：faces_user/人员名称/
        save_dir = os.path.join('./faces_user', person_name)

        # 如果目录已存在，先删除所有图片
        if os.path.exists(save_dir):
            for file in os.listdir(save_dir):
                file_path = os.path.join(save_dir, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    os.remove(file_path)
            if self.on_info_update:
                self.on_info_update(f"已清空 {person_name} 的旧照片")
        else:
            os.makedirs(save_dir, exist_ok=True)

        if self.on_info_update:
            self.on_info_update(f"\n{'=' * 60}")
            self.on_info_update(f"开始采集 [{person_name}] 的人脸数据")
            self.on_info_update(f"目标数量: {target_count}张（含3倍增强）")
            self.on_info_update(f"保存目录: {save_dir}")
            self.on_info_update(f"{'=' * 60}")

        # 检查已有图片数量（应该为0）
        existing_files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))]
        saved_count = len(existing_files)
        if self.on_info_update:
            self.on_info_update(f"📁 当前目录图片数: {saved_count} 张")

        frame_skip = 1  # 每1帧采集一次，提高采集速度
        frame_counter = 0

        return target_count, save_dir, frame_skip, frame_counter
    
    def start_collection(self, person_name, target_count):
        """
        开始数据采集
        
        Args:
            person_name: 人员名称
            target_count: 目标数量
        """
        self.current_user = person_name
        self.target_count, self.save_dir, self.frame_skip, self.frame_counter = self.prepare_collection(
            person_name, target_count
        )
        
        # 检查已有图片数量
        existing_files = [f for f in os.listdir(self.save_dir) if f.endswith(('.jpg', '.png'))]
        self.collection_count = len(existing_files)
        
        self.is_collecting = True
        
        if self.on_info_update:
            self.on_info_update(f"开始采集用户 '{self.current_user}' 的人脸数据...")
            
        if self.on_progress_update:
            self.on_progress_update(self.collection_count, self.target_count)
    
    def process_frame(self, frame, faces):
        """
        处理视频帧，采集人脸数据
        
        Args:
            frame: 视频帧
            faces: 检测到的人脸列表
            
        Returns:
            tuple: (处理后的帧, 是否完成采集)
        """
        if not self.is_collecting or self.collection_count >= self.target_count:
            return frame, True
            
        for i, d in enumerate(faces):
            x1 = max(d.top(), 0)
            y1 = min(d.bottom(), frame.shape[0])
            x2 = max(d.left(), 0)
            y2 = min(d.right(), frame.shape[1])
            
            face_img = frame[x1:y1, x2:y2]
            if face_img.size > 0 and face_img.shape[0] > 20 and face_img.shape[1] > 20:  # 确保人脸足够大
                self.frame_counter += 1

                # 每frame_skip帧采集一次
                if self.frame_counter % self.frame_skip == 0 and self.collection_count < self.target_count:
                    face_resized = cv2.resize(face_img, (self.size, self.size))

                    # 生成时间戳作为文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    base_filename = f"{self.current_user}_{timestamp}"

                    # 保存原始图片
                    original_path = os.path.join(self.save_dir, f"{base_filename}_orig.jpg")
                    cv2.imwrite(original_path, face_resized)
                    self.collection_count += 1

                    # 自动生成3个增强版本
                    if self.collection_count < self.target_count:
                        augmentations = self.apply_augmentations(face_resized)
                        for i, aug_img in enumerate(augmentations):
                            if self.collection_count >= self.target_count:
                                break

                            aug_path = os.path.join(self.save_dir, f"{base_filename}_aug{i + 1}.jpg")
                            cv2.imwrite(aug_path, aug_img)
                            self.collection_count += 1

                    # 更新进度
                    if self.on_progress_update:
                        self.on_progress_update(self.collection_count, self.target_count)

                    # 检查是否完成采集
                    if self.collection_count >= self.target_count:
                        self.stop_collection()
                        if self.on_collection_complete:
                            self.on_collection_complete(self.collection_count)
                        return frame, True

                # 绘制绿色采集框
                cv2.rectangle(frame, (x2, x1), (y2, y1), (0, 255, 0), 2)  # 绿色框

                # 只显示采集数字，不显示中文文字
                # 使用OpenCV的putText显示数字（数字不会乱码）
                text = f"{self.collection_count}/{self.target_count}"
                cv2.putText(frame, text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, False
    
    def stop_collection(self):
        """停止数据采集"""
        self.is_collecting = False
        if self.on_info_update:
            self.on_info_update(f"人脸采集停止，共保存 {self.collection_count} 张图片")
    
    def clear_user_data(self, person_name):
        """清除指定用户的数据"""
        save_dir = os.path.join('./faces_user', person_name)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            if self.on_info_update:
                self.on_info_update(f"已清除用户 {person_name} 的数据")
            return True
        return False
    
    def get_user_list(self):
        """获取已采集的用户列表"""
        faces_user_dir = './faces_user'
        if not os.path.exists(faces_user_dir):
            return []
        
        user_list = []
        for item in os.listdir(faces_user_dir):
            item_path = os.path.join(faces_user_dir, item)
            if os.path.isdir(item_path):
                # 统计该用户的图片数量
                image_files = [f for f in os.listdir(item_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                user_list.append({
                    'name': item,
                    'count': len(image_files),
                    'path': item_path
                })
        
        return user_list
    
    def check_data_balance(self):
        """检查数据平衡性"""
        user_list = self.get_user_list()
        if not user_list:
            return None
        
        counts = [user['count'] for user in user_list]
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        
        return {
            'user_count': len(user_list),
            'total_images': sum(counts),
            'min_images': min_count,
            'max_images': max_count,
            'avg_images': avg_count,
            'balance_ratio': min_count / max_count if max_count > 0 else 0,
            'user_details': user_list
        }
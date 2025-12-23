import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # 禁用TensorFlow 2.x行为
import os
import time
from PIL import Image, ImageTk
import threading
import shutil

from face_data_collector_tool import FaceDataCollectorTool  # 导入数据采集工具类
from face_recognition_tool import FaceRecognitionTool, FaceDetector  # 导入人脸识别工具类


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统 - 模块化版本")
        self.root.geometry("1200x800")

        # 初始化变量
        self.cap = None
        self.is_running = False
        self.camera_index = 0

        # 初始化人脸检测器
        self.face_detector = FaceDetector()

        # 初始化人脸采集器工具类
        self.face_collector_tool = FaceDataCollectorTool()

        # 初始化人脸识别器工具类
        self.face_recognition_tool = FaceRecognitionTool()

        # 设置回调函数
        self.face_collector_tool.on_progress_update = self._on_collection_progress
        self.face_collector_tool.on_collection_complete = self._on_collection_complete
        self.face_collector_tool.on_info_update = self.update_info

        self.face_recognition_tool.on_recognition_result = self._on_recognition_result
        self.face_recognition_tool.on_info_update = self.update_info

        # 检查项目目录结构
        self.check_project_structure()

        # 尝试加载模型
        self.model_loaded = self.face_recognition_tool.load_model()

        # 创建界面
        self.create_widgets()

        # 立即启动摄像头
        self.start_default_camera()

    def check_project_structure(self):
        """检查项目目录结构"""
        print("检查项目目录结构...")
        dirs_to_check = ['faces_user', 'faces_stranger', 'model_multi_class']
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
        left_frame = ttk.Frame(main_frame, width=720)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left_frame.pack_propagate(False)

        # 视频显示区域
        video_frame = ttk.LabelFrame(left_frame, text="实时画面", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame, background="black", text="摄像头已启动")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 右侧：控制面板和信息区域 (占40%宽度)
        right_frame = ttk.Frame(main_frame, width=480)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)

        # 模型状态
        model_status_frame = ttk.LabelFrame(right_frame, text="模型状态", padding=5)
        model_status_frame.pack(fill=tk.X, pady=(10, 10))

        # 创建模型状态标签
        self.model_status_label = ttk.Label(model_status_frame, font=("Arial", 9))
        self.model_status_label.pack(anchor=tk.W)

        # 更新模型状态显示
        self.update_model_status()

        # 控制面板
        control_frame = ttk.LabelFrame(right_frame, text="控制面板", padding=5)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 用户名输入
        username_frame = ttk.Frame(control_frame)
        username_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(username_frame, text="用户名:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.user_entry = ttk.Entry(username_frame, width=15, font=("Arial", 9))
        self.user_entry.insert(0, "default_user")
        self.user_entry.pack(side=tk.RIGHT)

        # 采集数量输入
        count_frame = ttk.Frame(control_frame)
        count_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(count_frame, text="采集数量:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.count_entry = ttk.Entry(count_frame, width=15, font=("Arial", 9))
        self.count_entry.insert(0, "500")
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
        self.progress['maximum'] = 500
        self.progress.pack(pady=(5, 0), fill=tk.X)
        self.progress.pack_forget()

        # 训练进度条
        self.train_progress = ttk.Progressbar(control_frame, mode='indeterminate', length=200)
        self.train_progress.pack(pady=(5, 0), fill=tk.X)
        self.train_progress.pack_forget()

        # 日志信息区域
        log_frame = ttk.LabelFrame(right_frame, text="日志信息", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.info_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Arial", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def update_model_status(self):
        """更新模型状态显示"""
        if self.model_loaded:
            status_text = f"✓ 模型加载成功!\n- 类别数: {self.face_recognition_tool.num_classes}\n- 类别名称: {self.face_recognition_tool.class_names}"
            self.model_status_label.config(text=status_text, foreground="green")
        else:
            status_text = "✗ 未检测到训练好的模型\n💡 请先采集人脸数据并训练模型"
            self.model_status_label.config(text=status_text, foreground="red")

    def toggle_collection(self):
        """切换采集模式"""
        if not self.is_running:
            messagebox.showwarning("警告", "摄像头未启动")
            return

        if not self.face_collector_tool.is_collecting:
            # 开始采集
            self.start_collection()
        else:
            # 停止采集
            self.stop_collection()

    def toggle_recognition(self):
        """切换识别模式"""
        if not self.is_running:
            messagebox.showwarning("警告", "摄像头未启动")
            return

        if self.face_collector_tool.is_collecting:
            self.stop_collection()

        # 切换识别状态
        if self.is_running and not self.face_collector_tool.is_collecting and self.model_loaded:
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
        """开始采集"""
        current_user = self.user_entry.get().strip()
        if not current_user:
            current_user = "default_user"

        # 获取采集数量
        try:
            count = int(self.count_entry.get())
            if count < 20:
                count = 20
            max_collection = count
        except:
            max_collection = 500

        # 获取摄像头索引
        if self.camera_var.get() == "外接(1)":
            new_camera_index = 1
        else:
            new_camera_index = 0

        # 如果摄像头索引改变，才重新打开摄像头
        if new_camera_index != self.camera_index:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(new_camera_index)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开摄像头")
                return
            self.camera_index = new_camera_index

        # 使用工具类开始采集
        try:
            self.face_collector_tool.start_collection(current_user, max_collection)
            self.collect_btn.config(text="停止采集")
            self.recognize_btn.config(state=tk.DISABLED)
            self.progress.pack(pady=(5, 0), fill=tk.X)
            self.progress['maximum'] = max_collection
            self.progress['value'] = self.face_collector_tool.collection_count
        except Exception as e:
            messagebox.showerror("错误", f"初始化采集失败: {e}")

    def _on_collection_progress(self, current_count, target_count):
        """采集进度更新回调"""
        self.progress['value'] = current_count
        self.status_label.config(text=f"状态: 正在采集人脸 ({current_count}/{target_count})")

    def _on_collection_complete(self, total_collected):
        """采集完成回调"""
        self.stop_collection()
        # 自动开始训练模型
        self.start_training()

    def _on_recognition_result(self, person_name, confidence):
        """识别结果回调"""
        info_msg = f"识别结果: {person_name}, 置信度: {confidence:.3f}"
        self.update_info(info_msg)

    def stop_collection(self):
        """停止采集"""
        self.face_collector_tool.stop_collection()
        self.collect_btn.config(text="人脸采集")
        self.recognize_btn.config(state=tk.NORMAL if self.model_loaded else tk.DISABLED)
        self.progress.pack_forget()

    def start_training(self):
        """开始模型训练"""
        # 检查是否有足够的数据
        faces_user_dir = 'faces_user'
        if not os.path.exists(faces_user_dir):
            messagebox.showerror("错误", "faces_user目录不存在")
            return

        class_names = [d for d in os.listdir(faces_user_dir) if os.path.isdir(os.path.join(faces_user_dir, d))]
        if len(class_names) < 2:
            messagebox.showerror("错误", "至少需要2个类别才能训练模型")
            return

        # 检查当前是否有采集在运行
        if self.face_collector_tool.is_collecting:
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
            # 使用训练工具类进行训练
            from face_train_tool import FaceModelTrainer

            trainer = FaceModelTrainer(
                faces_user_dir='faces_user',
                faces_stranger_dir='faces_stranger',
                model_dir='./model_multi_class',
                size=64,
                batch_size=32,
                learning_rate=0.001,
                target_samples_per_class=400,
                num_epochs=100,
                patience=10
            )

            success = trainer.train()
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

            # 快速重启流程
            self.update_info("正在重新加载模型...")

            # 关键修复：先关闭旧的识别器会话
            if self.face_recognition_tool:
                try:
                    self.face_recognition_tool.close()
                    self.face_recognition_tool.sess = None
                except:
                    pass

            # 重置TensorFlow的默认图
            tf.reset_default_graph()

            # 创建新的识别器实例
            self.face_recognition_tool = FaceRecognitionTool()

            # 尝试加载模型
            self.model_loaded = self.face_recognition_tool.load_model()

            # 更新模型状态显示
            self.update_model_status()

            if self.model_loaded:
                self.face_recognition_tool.on_recognition_result = self._on_recognition_result
                self.face_recognition_tool.on_info_update = self.update_info

                self.recognize_btn.config(state=tk.NORMAL)

                # 立即开始人脸识别
                self.is_running = True
                self.recognize_btn.config(text="停止识别")
                self.status_label.config(text="状态: 正在识别人脸")
                self.update_info("训练完成，立即开始人脸识别...")

                # 确保视频更新继续运行
                if not self.face_collector_tool.is_collecting:
                    self.update_video()
            else:
                self.update_info("模型加载失败，请重新启动程序")
        else:
            self.update_info("模型训练失败")
            self.status_label.config(text="状态: 模型训练失败")
            messagebox.showerror("错误", "模型训练失败，请检查faces_user目录中的数据")

            # 重新启用识别按钮
            self.recognize_btn.config(state=tk.NORMAL if self.model_loaded else tk.DISABLED)

    def stop_all(self):
        """停止所有操作但保持摄像头运行"""
        self.is_running = False
        self.face_collector_tool.stop_collection()
        self.recognize_btn.config(text="人脸识别", state=tk.NORMAL if self.model_loaded else tk.DISABLED)
        self.collect_btn.config(text="人脸采集", state=tk.NORMAL)
        self.progress.pack_forget()
        self.train_progress.stop()
        self.train_progress.pack_forget()
        self.status_label.config(text="状态: 摄像头已停止")
        self.update_info("摄像头已停止")

    def update_video(self):
        """更新视频帧"""
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(100, self.update_video)
            return

        # 检测人脸
        faces = self.face_detector.detect_faces(frame)

        # 如果是采集模式
        if self.face_collector_tool.is_collecting:
            frame, collection_complete = self.face_collector_tool.process_frame(frame, faces)
            if collection_complete:
                pass
        # 如果是识别模式
        elif self.is_running and not self.face_collector_tool.is_collecting and self.model_loaded:
            frame, recognition_results = self.face_recognition_tool.process_recognition(frame, faces)
        # 普通显示模式
        else:
            frame = self.face_detector.draw_faces(frame, faces, color=(255, 255, 0), thickness=2)

        # 转换为PIL图像并显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_pil = img_pil.resize((700, 500), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.video_label.img_tk = img_tk
        self.video_label.configure(image=img_tk)

        # 每10毫秒更新一次
        if self.is_running or self.face_collector_tool.is_collecting:
            self.root.after(10, self.update_video)

    def update_info(self, message):
        """更新信息显示区域"""
        current_time = time.strftime("%H:%M:%S", time.localtime())
        self.info_text.insert(tk.END, f"[{current_time}] {message}\n")
        self.info_text.see(tk.END)

    def close_app(self):
        """关闭应用程序"""
        # 关闭摄像头
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # 关闭识别器
        if hasattr(self, 'face_recognition_tool'):
            self.face_recognition_tool.close()
        # 关闭采集器
        if hasattr(self, 'face_collector_tool'):
            self.face_collector_tool.stop_collection()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()


if __name__ == "__main__":
    main()
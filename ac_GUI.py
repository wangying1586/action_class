import sys
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import cv2
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QPixmap, QImage
from model import TripletModel
from PyQt5.QtCore import Qt

class ActionRecognize():
    def __init__(self, ckpt_path="best_model_3.pth", device="cpu"):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.class_names = ['chuandi_keyiwupin', 'jian_keyiwupin', 'jiaotoujieer', 'jushou', 'normal', 'qili',
                            'shouzhuoxia_maitou', 'xianghoupiantou', 'zuoyoupiantou']

        model = TripletModel(num_classes=len(self.class_names))
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        self.model = model.to(device)

    def run(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb).convert('RGB')
        image = transforms.ToTensor()(image)
        image = self.transform(image)
        image = image.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            anchor_triplet_features = self.model(image.to(self.device))
            class_logits = self.model.classification_head(anchor_triplet_features)
            probabilities = F.softmax(class_logits, dim=1)
            preds = torch.argmax(class_logits, dim=1)

            predicted_class_index = preds.cpu()
            predicted_class_name = self.class_names[predicted_class_index]

            return predicted_class_name, probabilities


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.anomaly_results = {}  # 用于存储每个异常类别的结果（图片、类别标签等）
        self.class_names = ['chuandi_keyiwupin', 'jian_keyiwupin', 'jiaotoujieer', 'jushou', 'normal', 'qili',
                            'shouzhuoxia_maitou', 'xianghoupiantou', 'zuoyoupiantou']
        self.name = ['chuandi_keyiwupin', 'jian_keyiwupin', 'jiaotoujieer', 'jushou', 'qili',
                            'shouzhuoxia_maitou', 'xianghoupiantou', 'zuoyoupiantou']
        for name in self.name:
            self.anomaly_results[name] = {'img': None, 'label': None}
        print(self.anomaly_results)

        self.class_name_mapping = {
            'chuandi_keyiwupin': '传递可疑物品',
            'jian_keyiwupin': '捡可疑物品',
            'jiaotoujieer': '交头接耳',
            'jushou': '举手',
            'qili': '起立',
            'shouzhuoxia_maitou': '手放桌下并埋头',
            'xianghoupiantou': '向后偏头',
            'zuoyoupiantou': '左右偏头'
        }

        self.init_ui()

    def init_ui(self):
        # 整体布局为水平布局
        main_layout = QHBoxLayout()

        """视频播放 放在左边布局里 如下"""
        # 左边布局（垂直布局，留空）
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)
        """视频播放 放在左边布局里 如上"""

        # 右边布局（垂直布局）
        right_layout = QVBoxLayout()

        # 第一行布局（水平布局，包含4个异常标签，后续替换为图片）
        row1_layout = QHBoxLayout()
        for i in range(1, 5):
            label = QLabel()
            blank_pixmap = QPixmap(100, 100)
            blank_pixmap.fill(Qt.white)
            label.setPixmap(blank_pixmap)
            row1_layout.addWidget(label)
            self.anomaly_results[self.name[i - 1]]["label"] = label
        right_layout.addLayout(row1_layout)

        # 第二行布局（水平布局，包含4个动作标签）
        row2_layout = QHBoxLayout()
        actions = ["传递可疑物品", "捡可疑物品", "交头接耳", "举手"]
        self.action_labels = {}
        for action in actions:
            label = QLabel(action)
            self.action_labels[action] = label
            row2_layout.addWidget(label)
        right_layout.addLayout(row2_layout)

        # 第三行布局（水平布局，包含4个异常标签，后续替换为图片）
        row3_layout = QHBoxLayout()
        for i in range(5, 9):
            label = QLabel()
            blank_pixmap = QPixmap(100, 100)
            blank_pixmap.fill(Qt.white)
            label.setPixmap(blank_pixmap)
            row3_layout.addWidget(label)
            self.anomaly_results[self.name[i - 1]]["label"] = label
        right_layout.addLayout(row3_layout)

        # 第四行布局（水平布局，包含4个动作标签）
        row4_layout = QHBoxLayout()
        actions = ["起立", "手放桌下并埋头", "向后偏头", "左右偏头"]
        for action in actions:
            label = QLabel(action)
            self.action_labels[action] = label
            row4_layout.addWidget(label)
        right_layout.addLayout(row4_layout)

        # 检测.log信息输出和时间点，异常类别标签
        self.log_label = QLabel("检测.log信息输出")
        self.log_scroll_area = QScrollArea()
        self.log_scroll_area.setWidget(self.log_label)
        self.log_scroll_area.setWidgetResizable(True)
        self.log_scroll_area.setFixedSize(300, 200)  # 这里设置固定大小，可根据需求调整

        # 设置QScrollArea在水平方向上的拉伸策略
        right_layout.addWidget(self.log_scroll_area, stretch=1, alignment=Qt.AlignHCenter)

        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def show_results(self, img_paths):
        action_recognize = ActionRecognize()
        log_text = ""
        for img_path in img_paths:
            img_bgr = cv2.imread(img_path)
            predicted_class_name, probabilities = action_recognize.run(img_bgr)

            if predicted_class_name!= 'normal':
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                try:
                    # 假设 img_rgb 是从 cv2 读取并转换后的图片数据（numpy数组形式）
                    height, width, channel = img_rgb.shape
                    aspect_ratio = width / height
                    new_width = 100
                    new_height = int(100 / aspect_ratio)
                    qimg = QImage(img_rgb.data, width, height, 3 * width, QImage.Format_RGB888)
                    img_pixmap = QPixmap.fromImage(qimg)
                    img_pixmap = img_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)

                    # 根据类别映射到对应的异常标签位置
                    if predicted_class_name in self.anomaly_results:
                        print(predicted_class_name)
                        label = self.anomaly_results[predicted_class_name]['label']
                        # print(label)
                        label.setPixmap(img_pixmap)

                    probability_value = probabilities[0][self.class_names.index(predicted_class_name)].item()
                    # print(probabilities[0])
                    # print(self.name.index(predicted_class_name))
                    img_name = img_path.split("\\")[-1]
                    log_text += f"图片: {img_name}, 检测到的类别: {self.class_name_mapping[predicted_class_name]}, 概率: {probability_value:.4f}\n"

                except Exception as e:
                    print(f"Error in QPixmap operation: {e}")

        self.log_label.setText(log_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    test_img_names = os.listdir("pipeline_test_imgs")
    test_img_pathes = [os.path.join("pipeline_test_imgs", name) for name in test_img_names]
    window.show_results(test_img_pathes)
    window.show()
    sys.exit(app.exec_())
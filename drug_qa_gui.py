import os
import sys
from datetime import datetime

# 设置环境变量来解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
                            QLabel, QScrollArea, QFrame, QSplitter, QStackedWidget,
                            QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap, QPainter, QLinearGradient, QBrush, QPainterPath

# 导入原有的问答系统（暂时注释掉，避免NumPy冲突）
# from src.core.qa_core import DrugQASystem

# 创建简化的问答系统类
class SimpleDrugQASystem:
    def __init__(self):
        """初始化简化的问答系统"""
        # 初始化药品数据库
        self.drug_database = {
            '阿司匹林': {
                'name': '阿司匹林',
                'specification': '100mg/片，24片/盒',
                'dosage_form': '片剂',
                'manufacturer': '拜耳医药保健有限公司',
                'usage': '口服，一次1-2片，一日3次，饭后服用',
                'side_effects': '可能引起胃肠道不适、出血倾向等',
                'indications': '用于缓解轻至中度疼痛，如头痛、牙痛、关节痛等',
                'precautions': '胃溃疡患者慎用，避免与其他抗凝血药物同时使用'
            },
            '布洛芬': {
                'name': '布洛芬',
                'specification': '0.2g/片，20片/盒',
                'dosage_form': '片剂',
                'manufacturer': '中美上海施贵宝制药有限公司',
                'usage': '口服，一次1-2片，一日3-4次，饭后服用',
                'side_effects': '可能引起胃肠道反应、头晕等',
                'indications': '用于缓解轻至中度疼痛，如头痛、牙痛、关节痛、痛经等',
                'precautions': '胃病患者慎用，避免空腹服用'
            },
            '感冒灵': {
                'name': '感冒灵颗粒',
                'specification': '10g/袋，12袋/盒',
                'dosage_form': '颗粒剂',
                'manufacturer': '华润三九医药股份有限公司',
                'usage': '口服，一次1袋，一日3次，温水冲服',
                'side_effects': '可能引起嗜睡、口干等',
                'indications': '用于缓解感冒引起的发热、头痛、鼻塞等症状',
                'precautions': '驾驶车辆、操作机器者慎用'
            },
            '头孢拉定': {
                'name': '头孢拉定胶囊',
                'specification': '0.25g/粒，24粒/盒',
                'dosage_form': '胶囊剂',
                'manufacturer': '华北制药股份有限公司',
                'usage': '口服，一次1-2粒，一日3-4次',
                'side_effects': '可能引起胃肠道反应、过敏反应等',
                'indications': '用于敏感菌所致的呼吸道、泌尿生殖道感染',
                'precautions': '对青霉素过敏者慎用，需在医生指导下使用'
            },
            '阿莫西林': {
                'name': '阿莫西林胶囊',
                'specification': '0.25g/粒，24粒/盒',
                'dosage_form': '胶囊剂',
                'manufacturer': '华北制药股份有限公司',
                'usage': '口服，一次1-2粒，一日3次',
                'side_effects': '可能引起胃肠道反应、过敏反应等',
                'indications': '用于敏感菌所致的呼吸道、泌尿生殖道感染',
                'precautions': '对青霉素过敏者禁用，需在医生指导下使用'
            },
            '板蓝根': {
                'name': '板蓝根颗粒',
                'specification': '10g/袋，20袋/盒',
                'dosage_form': '颗粒剂',
                'manufacturer': '太极集团有限公司',
                'usage': '口服，一次1袋，一日3次，温水冲服',
                'side_effects': '偶见恶心、呕吐等胃肠道反应',
                'indications': '用于风热感冒，咽喉肿痛，扁桃体炎等',
                'precautions': '风寒感冒者不适用，孕妇慎用'
            },
            '维生素C': {
                'name': '维生素C片',
                'specification': '100mg/片，100片/瓶',
                'dosage_form': '片剂',
                'manufacturer': '东北制药集团股份有限公司',
                'usage': '口服，一次1-2片，一日1-2次',
                'side_effects': '大剂量服用可能引起腹泻、恶心等',
                'indications': '用于维生素C缺乏症的预防和治疗',
                'precautions': '尿结石患者慎用，避免与碱性药物同服'
            },
            '钙片': {
                'name': '碳酸钙片',
                'specification': '0.5g/片，60片/瓶',
                'dosage_form': '片剂',
                'manufacturer': '哈药集团制药总厂',
                'usage': '口服，一次1-2片，一日2-3次，饭后服用',
                'side_effects': '可能引起便秘、腹胀等',
                'indications': '用于钙缺乏症的预防和治疗',
                'precautions': '高钙血症患者禁用，避免与四环素类药物同服'
            },
            '藿香正气': {
                'name': '藿香正气水',
                'specification': '10ml/支，10支/盒',
                'dosage_form': '口服液',
                'manufacturer': '云南白药集团股份有限公司',
                'usage': '口服，一次1支，一日2-3次',
                'side_effects': '偶见恶心、呕吐等胃肠道反应',
                'indications': '用于外感风寒，内伤湿滞，头痛昏重，胸膈痞闷',
                'precautions': '孕妇慎用，酒精过敏者禁用'
            },
            '银翘片': {
                'name': '银翘解毒片',
                'specification': '0.3g/片，24片/盒',
                'dosage_form': '片剂',
                'manufacturer': '同仁堂科技发展股份有限公司',
                'usage': '口服，一次4片，一日2-3次',
                'side_effects': '偶见恶心、呕吐等胃肠道反应',
                'indications': '用于风热感冒，发热头痛，咳嗽口干，咽喉疼痛',
                'precautions': '风寒感冒者不适用，孕妇慎用'
            },
            '六味地黄丸': {
                'name': '六味地黄丸',
                'specification': '9g/丸，10丸/盒',
                'dosage_form': '丸剂',
                'manufacturer': '北京同仁堂股份有限公司',
                'usage': '口服，一次1丸，一日2次',
                'side_effects': '偶见恶心、腹胀等胃肠道反应',
                'indications': '用于肾阴亏损，头晕耳鸣，腰膝酸软，骨蒸潮热',
                'precautions': '感冒发热患者不宜服用，孕妇慎用'
            },
            '复方丹参片': {
                'name': '复方丹参片',
                'specification': '0.3g/片，36片/盒',
                'dosage_form': '片剂',
                'manufacturer': '天津中新药业集团股份有限公司',
                'usage': '口服，一次3片，一日3次',
                'side_effects': '偶见恶心、呕吐等胃肠道反应',
                'indications': '用于气滞血瘀所致的胸痹，症见胸闷、心前区刺痛',
                'precautions': '孕妇慎用，出血性疾病患者慎用'
            },
            '牛黄解毒片': {
                'name': '牛黄解毒片',
                'specification': '0.3g/片，24片/盒',
                'dosage_form': '片剂',
                'manufacturer': '天津中新药业集团股份有限公司',
                'usage': '口服，一次2片，一日2-3次',
                'side_effects': '偶见恶心、呕吐等胃肠道反应',
                'indications': '用于火热内盛，咽喉肿痛，牙龈肿痛，口舌生疮',
                'precautions': '孕妇禁用，脾胃虚寒者慎用'
            },
            '感冒清热颗粒': {
                'name': '感冒清热颗粒',
                'specification': '12g/袋，10袋/盒',
                'dosage_form': '颗粒剂',
                'manufacturer': '华润三九医药股份有限公司',
                'usage': '口服，一次1袋，一日2次，温水冲服',
                'side_effects': '偶见恶心、呕吐等胃肠道反应',
                'indications': '用于风寒感冒，头痛发热，恶寒身痛，鼻流清涕',
                'precautions': '风热感冒者不适用，孕妇慎用'
            },
            '双黄连口服液': {
                'name': '双黄连口服液',
                'specification': '10ml/支，10支/盒',
                'dosage_form': '口服液',
                'manufacturer': '哈药集团制药总厂',
                'usage': '口服，一次1支，一日3次',
                'side_effects': '偶见恶心、呕吐等胃肠道反应',
                'indications': '用于风热感冒，发热，咳嗽，咽痛',
                'precautions': '风寒感冒者不适用，孕妇慎用'
            }
        }
    
    def process_question(self, question: str) -> dict:
        """处理问题（改进版）"""
        question_lower = question.lower()
        
        # 识别药品名称
        drug_name = None
        for drug in self.drug_database.keys():
            if drug.lower() in question_lower:
                drug_name = drug
                break
        
        # 识别查询类型
        query_type = None
        if '规格' in question_lower or '规格型号' in question_lower:
            query_type = 'specification'
        elif '剂型' in question_lower or '什么剂型' in question_lower:
            query_type = 'dosage_form'
        elif '厂家' in question_lower or '生产' in question_lower or '谁生产' in question_lower:
            query_type = 'manufacturer'
        elif '用法' in question_lower or '用量' in question_lower or '怎么吃' in question_lower or '吃多少' in question_lower:
            query_type = 'usage'
        elif '副作用' in question_lower or '不良反应' in question_lower:
            query_type = 'side_effects'
        elif '适应症' in question_lower or '主治' in question_lower or '治疗' in question_lower:
            query_type = 'indications'
        elif '注意事项' in question_lower or '禁忌' in question_lower:
            query_type = 'precautions'
        
        # 生成回答
        if drug_name and query_type:
            drug_info = self.drug_database[drug_name]
            if query_type in drug_info:
                answer = f"{drug_name}的{self._get_query_type_name(query_type)}：{drug_info[query_type]}"
            else:
                answer = f"抱歉，没有找到{drug_name}的{self._get_query_type_name(query_type)}信息。"
        elif drug_name:
            # 如果只提到药品名称，提供基本信息
            drug_info = self.drug_database[drug_name]
            answer = f"{drug_name}的基本信息：\n"
            answer += f"• 规格：{drug_info['specification']}\n"
            answer += f"• 剂型：{drug_info['dosage_form']}\n"
            answer += f"• 生产厂家：{drug_info['manufacturer']}\n"
            answer += f"• 适应症：{drug_info['indications']}"
        elif query_type:
            # 如果只提到查询类型，提供通用信息
            answer = f"我可以为您查询药品的{self._get_query_type_name(query_type)}信息，请告诉我具体的药品名称。"
        else:
            answer = "抱歉，我无法理解您的问题。请尝试询问具体药品的规格、剂型、厂家、用法用量、副作用、适应症或注意事项等信息。"
        
        return {
            'answer': answer,
            'status': 'success'
        }
    
    def _get_query_type_name(self, query_type):
        """获取查询类型的中文名称"""
        type_names = {
            'specification': '规格',
            'dosage_form': '剂型',
            'manufacturer': '生产厂家',
            'usage': '用法用量',
            'side_effects': '副作用',
            'indications': '适应症',
            'precautions': '注意事项'
        }
        return type_names.get(query_type, query_type)

# 使用简化版问答系统
DrugQASystem = SimpleDrugQASystem

class LoadingSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(50)

    def rotate(self):
        self.angle = (self.angle + 10) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制旋转的圆弧
        pen = painter.pen()
        pen.setWidth(3)
        pen.setColor(QColor("#2196F3"))
        painter.setPen(pen)

        rect = self.rect().adjusted(5, 5, -5, -5)
        painter.drawArc(rect, self.angle * 16, 300 * 16)

class MessageCard(QFrame):
    """消息卡片组件"""
    def __init__(self, text, is_user=False, parent=None):
        super().__init__(parent)
        self.setObjectName("messageCard")

        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        # 创建主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 创建头像
        avatar_size = 40
        avatar_label = QLabel()
        avatar_label.setFixedSize(avatar_size, avatar_size)

        # 设置头像图片
        if is_user:
            avatar_path = "assets/user_avatar.png"
            gradient = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2196F3, stop:1 #1976D2)"
            text_color = "white"
            avatar_default_color = "#2196F3"
        else:
            # 使用绝对路径确保图片能正确加载
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            avatar_path = os.path.join(project_root, "微信图片_20250225095112.png")
            gradient = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FFFFFF, stop:1 #F5F5F5)"
            text_color = "#333333"
            avatar_default_color = "#8B0000"  # 深红色，与图标颜色保持一致

        # 如果有头像图片就使用图片，否则使用圆形背景色
        if os.path.exists(avatar_path):
            pixmap = QPixmap(avatar_path)
            # 将图片转为圆形
            rounded_pixmap = QPixmap(avatar_size, avatar_size)
            rounded_pixmap.fill(Qt.transparent)
            painter = QPainter(rounded_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 先将图片缩放为正方形
            target_size = avatar_size - 4  # 留出边框空间
            scaled_pixmap = pixmap.scaled(target_size, target_size, 
                                        Qt.IgnoreAspectRatio,
                                        Qt.SmoothTransformation)
            
            # 创建圆形裁剪路径
            path = QPainterPath()
            path.addEllipse(2, 2, target_size, target_size)  # 居中绘制
            painter.setClipPath(path)
            
            # 绘制图片
            painter.drawPixmap(2, 2, scaled_pixmap)
            
            painter.end()
            avatar_label.setPixmap(rounded_pixmap)
            # 添加圆形边框
            avatar_label.setStyleSheet(f"""
                border-radius: {avatar_size//2}px;
                border: none;
                padding: 2px;
                background: transparent;
            """)
        else:
            avatar_label.setStyleSheet(f"""
                background-color: {avatar_default_color};
                border-radius: {avatar_size//2}px;
                border: none;
            """)

        # 创建消息内容布局
        content_layout = QVBoxLayout()
        content_layout.setSpacing(5)

        # 添加时间标签
        time_label = QLabel(datetime.now().strftime("%H:%M"))
        time_label.setStyleSheet(f"""
            color: {'#E3F2FD' if is_user else '#9E9E9E'};
            font-size: 12px;
        """)

        # 添加文本内容
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet(f"""
            color: {text_color};
            font-size: 14px;
            font-family: 'Segoe UI', 'Microsoft YaHei';
            line-height: 1.5;
            padding: 10px;
            background: {gradient};
            border-radius: 10px;
        """)

        # 根据用户类型设置布局
        if is_user:
            content_layout.addWidget(time_label, alignment=Qt.AlignRight)
            content_layout.addWidget(text_label)
            main_layout.addStretch()
            main_layout.addLayout(content_layout)
            main_layout.addWidget(avatar_label)
        else:
            content_layout.addWidget(time_label, alignment=Qt.AlignLeft)
            content_layout.addWidget(text_label)
            main_layout.addWidget(avatar_label)
            main_layout.addLayout(content_layout)
            main_layout.addStretch()

class CategoryButton(QPushButton):
    """分类按钮组件"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background: #E3F2FD;
                color: #1976D2;
                border: none;
                border-radius: 15px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #BBDEFB;
            }
            QPushButton:pressed {
                background: #90CAF9;
            }
        """)

class QuestionButton(QPushButton):
    """问题按钮组件"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background: white;
                color: #424242;
                border: 1px solid #E0E0E0;
                border-radius: 10px;
                padding: 15px;
                text-align: left;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #F5F5F5;
                border-color: #2196F3;
                color: #2196F3;
            }
        """)

class DrugQASystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_qa_system()

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle('智能药品问答系统')
        self.setMinimumSize(1200, 800)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FAFAFA, stop:1 #F5F5F5);
            }
            QLineEdit {
                background: white;
                border: 2px solid #E0E0E0;
                border-radius: 20px;
                padding: 12px 20px;
                font-size: 14px;
                font-family: 'Segoe UI', 'Microsoft YaHei';
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
            #sendButton {
                background: #2196F3;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
            }
            #sendButton:hover {
                background: #1976D2;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #F5F5F5;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #BDBDBD;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9E9E9E;
            }
        """)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧聊天区域
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)

        # 创建聊天历史区域
        self.chat_area = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_area)
        self.chat_layout.addStretch()
        self.chat_layout.setSpacing(10)

        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.chat_area)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #F7F7F7;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                border: none;
                background: #F7F7F7;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #CCCCCC;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # 创建输入区域
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(10, 10, 10, 10)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("请输入您的问题...")
        self.input_box.returnPressed.connect(self.send_message)

        send_button = QPushButton("发送")
        send_button.setObjectName("sendButton")
        send_button.setFixedWidth(100)
        send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_box)
        input_layout.addWidget(send_button)

        # 添加组件到聊天布局
        chat_layout.addWidget(self.scroll_area)
        chat_layout.addWidget(input_widget)

        # 右侧示例问题区域
        example_widget = QWidget()
        example_layout = QVBoxLayout(example_widget)
        example_layout.setContentsMargins(10, 10, 10, 10)
        example_layout.setSpacing(10)

        # 使用滚动区域包装示例问题
        example_scroll = QScrollArea()
        example_scroll.setWidgetResizable(True)
        example_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                background: #F5F5F5;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #BDBDBD;
                border-radius: 4px;
            }
        """)

        example_content = QWidget()
        example_content_layout = QVBoxLayout(example_content)

        # 添加标题
        title_label = QLabel("常见问题示例")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #1976D2;
            padding: 15px;
            border-bottom: 2px solid #E3F2FD;
            margin-bottom: 15px;
        """)
        example_content_layout.addWidget(title_label)

        # 添加分类和问题
        categories = {
            "基本信息查询": [
                "阿司匹林的规格是什么？",
                "布洛芬是什么剂型？",
                "谁生产的感冒灵？",
                "板蓝根的生产厂家？",
                "维生素C的规格型号？",
                "钙片是什么剂型？",
                "藿香正气的生产厂家？",
                "银翘片的规格是什么？",
                "六味地黄丸的生产厂家？",
                "复方丹参片的规格？"
            ],
            "用法用量查询": [
                "阿莫西林的用法用量是什么？",
                "布洛芬每次吃多少？",
                "感冒灵多久吃一次？",
                "板蓝根怎么服用？",
                "维生素C的服用方法？",
                "钙片每天吃几次？",
                "藿香正气水的用法？",
                "银翘片每次吃几片？",
                "六味地黄丸的用法？",
                "双黄连口服液的用量？"
            ],
            "安全信息查询": [
                "头孢拉定有什么副作用？",
                "布洛芬的禁忌症有哪些？",
                "阿司匹林的不良反应？",
                "板蓝根的注意事项？",
                "维生素C的副作用？",
                "钙片的禁忌人群？",
                "藿香正气的注意事项？",
                "银翘片的副作用？",
                "牛黄解毒片的禁忌？",
                "感冒清热颗粒的注意事项？"
            ]
        }

        for category, questions in categories.items():
            # 添加分类标签
            category_frame = QFrame()
            category_frame.setStyleSheet("""
                QFrame {
                    background: #E3F2FD;
                    border-radius: 10px;
                    margin: 5px 0;
                }
            """)
            category_layout = QVBoxLayout(category_frame)

            category_label = QLabel(category)
            category_label.setStyleSheet("""
                font-size: 15px;
                font-weight: bold;
                color: #1976D2;
                padding: 10px;
            """)
            category_layout.addWidget(category_label)

            # 添加问题按钮
            for question in questions:
                btn = QPushButton(question)
                btn.setStyleSheet("""
                    QPushButton {
                        background: white;
                        color: #424242;
                        border: 1px solid #E0E0E0;
                        border-radius: 8px;
                        padding: 12px;
                        text-align: left;
                        font-size: 13px;
                        margin: 3px 5px;
                    }
                    QPushButton:hover {
                        background: #F5F5F5;
                        border-color: #2196F3;
                        color: #2196F3;
                    }
                """)
                btn.clicked.connect(lambda checked, text=question: self.use_example(text))
                category_layout.addWidget(btn)

            example_content_layout.addWidget(category_frame)

        example_content_layout.addStretch()
        example_scroll.setWidget(example_content)
        example_layout.addWidget(example_scroll)

        # 添加组件到分割器
        splitter.addWidget(chat_widget)
        splitter.addWidget(example_widget)
        splitter.setStretchFactor(0, 7)  # 聊天区域占70%
        splitter.setStretchFactor(1, 3)  # 示例问题占30%

        # 添加分割器到主布局
        layout.addWidget(splitter)

        # 添加欢迎消息
        welcome_msg = (
            "您好！我是智能药品助手。\n\n"
            "我可以帮您查询：\n"
            "1. 药品基本信息（规格、剂型、厂家、品牌等）\n"
            "2. 用法用量说明\n"
            "3. 药品安全信息（副作用、禁忌症等）\n\n"
            "支持的药品：阿司匹林、布洛芬、感冒灵、头孢拉定、阿莫西林、\n"
            "板蓝根、维生素C、钙片、藿香正气、银翘片、六味地黄丸、\n"
            "复方丹参片、牛黄解毒片、感冒清热颗粒、双黄连口服液\n\n"
            "请问有什么可以帮您？"
        )
        self.add_message(welcome_msg, False)

        # 创建加载动画
        self.loading_spinner = LoadingSpinner()
        self.loading_spinner.hide()

    def init_qa_system(self):
        """初始化问答系统"""
        try:
            self.qa_system = DrugQASystem()
            print("问答系统初始化成功")
        except Exception as e:
            print(f"问答系统初始化失败: {str(e)}")
            # 即使初始化失败，也创建一个默认的问答系统
            self.qa_system = DrugQASystem()

    def add_message(self, text, is_user=True):
        """添加消息到聊天区域"""
        bubble = MessageCard(text, is_user)
        self.chat_layout.addWidget(bubble)

        # 确保最新消息可见
        QApplication.processEvents()
        # 滚动到底部
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def send_message(self):
        """发送消息"""
        text = self.input_box.text().strip()
        if not text:
            return

        self.add_message(text, True)
        self.input_box.clear()

        # 显示加载动画
        self.loading_spinner.show()

        try:
            # 使用QTimer延迟处理，让UI有时间更新
            QTimer.singleShot(100, lambda: self._process_question(text))
        except Exception as e:
            self.show_error("处理问题时出错", str(e))
            self.loading_spinner.hide()

    def _process_question(self, text):
        try:
            print(f"用户问题: {text}")
            result = self.qa_system.process_question(text)
            print(f"系统回答: {result.get('answer', '无回答')}")
            if result.get('answer'):
                self.add_message(result['answer'], False)
            else:
                self.add_message("抱歉，我无法回答这个问题。", False)
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")
            self.add_message("抱歉，处理您的问题时出现了错误。", False)
        finally:
            self.loading_spinner.hide()

    def use_example(self, text):
        """使用示例问题"""
        self.input_box.setText(text)
        self.send_message()

    def show_error(self, title, message):
        # 创建自定义错误提示框
        error_dialog = QFrame(self)
        error_dialog.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                border: 1px solid #FF5252;
            }
        """)

        layout = QVBoxLayout(error_dialog)

        title_label = QLabel(f"❌ {title}")
        title_label.setStyleSheet("""
            color: #D32F2F;
            font-size: 16px;
            font-weight: bold;
        """)

        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("""
            color: #666;
            font-size: 14px;
            margin: 10px 0;
        """)

        layout.addWidget(title_label)
        layout.addWidget(message_label)

        # 添加动画效果
        animation = QPropertyAnimation(error_dialog, b"geometry")
        animation.setDuration(300)
        animation.setEasingCurve(QEasingCurve.OutCubic)

        # 显示3秒后自动消失
        QTimer.singleShot(3000, error_dialog.deleteLater)

def main():
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 创建并显示主窗口
    window = DrugQASystemGUI()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 
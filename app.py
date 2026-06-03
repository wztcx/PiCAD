# -*- coding: utf-8 -*-
"""
智能医学影像工作站：基于DenseNet-121与Grad-CAM的肺炎辅助诊断与病灶定位系统
运行命令：streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
# 引入新版隐藏权重枚举以防止警告
try:
    from torchvision.models import DenseNet121_Weights
    HAS_WEIGHTS = True
except ImportError:
    HAS_WEIGHTS = False

# ==========================================
# 1. 页面基本配置 (必须作为Streamlit首条命令)
# ==========================================
st.set_page_config(
    page_title="肺炎AI辅助诊断工作站",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 可解释性 AI (XAI) 核心渲染引擎
# ==========================================
class AdvancedGradCAMEngine:
    """
    Grad-CAM 核心引擎
    通过动态注册软件探针（Hook），异步拦截特征图与逆向梯度流，解算病灶归一化热力图
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None  # 暂存逆向梯度
        self.features = None   # 暂存正向特征图
        
        # 注册前向和反向探针
        self.forward_hook = self.target_layer.register_forward_hook(self._save_feature)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_feature(self, module, input, output):
        self.features = output
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def execute_dual_track_solving(self, input_tensor, category_idx=1):
        """
        双轨并行解算逻辑
        :param input_tensor: 标准张量化的胸片数据 [1, 3, 224, 224]
        :param category_idx: 目标类别，1代表肺炎 (Pneumonia)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # 轨道一：正向传播解算置信度
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
        
        # 轨道二：决策路径锁定，触发反向传播
        target_score = logits[0, category_idx]
        target_score.backward()
        
        # 提取探针截获的特征图与梯度流 (转换为NumPy进行后处理)
        grads_matrix = self.gradients.cpu().data.numpy()[0]
        features_matrix = self.features.cpu().data.numpy()[0]
        
        # 计算通道重要性权重 (对梯度执行全局平均池化 GAP)
        channel_weights = np.mean(grads_matrix, axis=(1, 2))
        
        # 加权求和合成类激活映射图 (CAM)
        cam_map = np.zeros(features_matrix.shape[1:], dtype=np.float32)
        for idx, w in enumerate(channel_weights):
            cam_map += w * features_matrix[idx, :, :]
            
        # 线性整流 (ReLU) 剔除负向归因，并执行最大值归一化
        cam_map = np.maximum(cam_map, 0)
        if np.max(cam_map) != 0:
            cam_map = cam_map / np.max(cam_map)
            
        return probabilities, cam_map

    def release_hooks(self):
        """显式释放探针，防止内存泄漏"""
        self.forward_hook.remove()
        self.backward_hook.remove()


# ==========================================
# 3. 骨干网络加载与缓存保护机制
# ==========================================
# @st.cache_resource
# def init_diagnostic_system():
#     """
#     初始化深度学习骨干网络并挂载Grad-CAM引擎
#     """
#     if HAS_WEIGHTS:
#         model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
#     else:
#         model = models.densenet121(pretrained=True)
        
#     num_ftrs = model.classifier.in_features
#     model.classifier = nn.Linear(num_ftrs, 2)
    
#     # ❌ 【删除或注释掉旧的这一行】
#     # target_layer = model.features.norm5
    
#     # 🌟 【替换为下面这一行：挂载到整个第四卷积块的输出端】
#     target_layer = model.features.denseblock4
    
#     # 实例化Grad-CAM解算器
#     cam_engine = AdvancedGradCAMEngine(model, target_layer)
#     return model, cam_engine
@st.cache_resource
def init_diagnostic_system():
    """
    初始化深度学习骨干网络并挂载Grad-CAM引擎（已整合本地 best_model.pth 真实权重）
    """
    # 1. 🌟 纯离线创建模型骨架（直接干掉之前的 if-else 联网加载逻辑，UI秒开）
    model = models.densenet121(weights=None)
        
    # 2. 重塑全连接层（必须和 train.py 里的二分类结构完全一致）
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    
    # 3. 🌟 核心注入：在这里强制加载你刚刚训练出来的真实肺炎诊断权重！
    # 💡 确保你的 best_model.pth 文件和这个 app.py 放在同一个文件夹里
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    
    # 4. 锁定第四卷积块作为探针挂载点（完美避开之前的 inplace 报错）
    target_layer = model.features.denseblock4
    
    # 5. 实例化Grad-CAM解算器
    cam_engine = AdvancedGradCAMEngine(model, target_layer)
    return model, cam_engine

# ==========================================
# 4. 工作站 UI 展现层与业务流控制
# ==========================================
def main():
    # 初始化后台算法舱
    model, cam_engine = init_diagnostic_system()
    
    # 界面大标题
    st.title("🔬 肺炎影像AI辅助诊断与多模态可解释性工作站")
    st.markdown("---")
    
    # ------------------------------------------
    # 业务流分区一：左侧边栏病历管理舱 (状态0 -> 状态1)
    # ------------------------------------------
    with st.sidebar:
        st.header("📋 患者病历档案管理")
        p_id = st.text_input("患者编号 (ID)", value="PNT-2026-0032")
        p_name = st.text_input("基本姓名", value="张伟")
        p_age = st.number_input("年龄 (岁)", min_value=0, max_value=120, value=45)
        p_gender = st.selectbox("性别", ["男", "女"])
        
        st.markdown("---")
        st.info("""
        **使用说明：**
        1. 在侧边栏录入患者基础体征档案。
        2. 在主界面左侧上传标准的胸部X线影像(CXR)。
        3. 系统将自动执行正反向双轨解算，输出诊断报告及病灶红外精确定位。
        """)

    # ------------------------------------------
    # 业务流分区二：主阅片区双栏拓扑布局
    # ------------------------------------------
    col_left, col_right = st.columns([1, 1])
    
    # 栏目一：数字胶片导入与动态预视区 (状态1 -> 状态2)
    with col_left:
        st.subheader("📤 第一步：导入胸部 X 线影像")
        uploaded_file = st.file_uploader(
            "支持拖拽或选择标准的CXR数字胶片 (格式: JPG, PNG, JPEG)", 
            type=["jpg", "png", "jpeg"]
        )
        
        if uploaded_file is not None:
            try:
                # 读取并渲染原始灰度影像
                raw_image = Image.open(uploaded_file).convert("RGB")
                st.image(raw_image, caption="当前加载的患者原始胸部 X 线片 (CXR)", use_container_width=True)
            except Exception as e:
                st.error(f"❌ 文件损坏或格式不兼容，防灾机制已拦截。错误详情: {e}")
                return
        else:
            # 状态0：初始化空载提示
            st.warning("⏳ 正在等待医生导入患者胸片数据...")
            
    # 栏目二：AI决策质控与可视化工作台 (状态2 -> 状态3 -> 状态4)
    with col_right:
        st.subheader("🔬 第二步：智能工作站 AI 分析结果")
        
        if uploaded_file is not None:
            # 1. 触发标准张量化流线
            preprocess_pipeline = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            # 扩展 Batch 维度 [3, 224, 224] -> [1, 3, 224, 224]
            input_tensor = preprocess_pipeline(raw_image).unsqueeze(0)
            
            # 2. 状态防抖拦截：调用沙漏动画并执行双轨解算
            with st.spinner("🚀 算法核心正在执行高密度前反向计算，请稍候..."):
                probs, cam_raw = cam_engine.execute_dual_track_solving(input_tensor, category_idx=1)
            
            # 3. 联动展现：量化概率透视区
            st.write("#### 🎯 辅助诊断置信度输出:")
            pneumonia_risk = probs[1] * 100
            st.progress(int(pneumonia_risk))
            st.write(f"▶️ **肺炎患病风险:** `{pneumonia_risk:.2f}%` ｜ **健康组织置信度:** `{probs[0]*100:.2f}%`")
            
            # 4. 后处理渲染：多模态空间对齐与双线性插值渲染
            # 将原始图像缩放到224x224以和热力图矩阵进行绝对像素对齐
            img_cv = cv2.resize(np.array(raw_image), (224, 224))
            
            # 空间双线性插值平滑上采样：7x7 -> 224x224
            cam_resized = cv2.resize(cam_raw, (224, 224))
            
            # 应用 COLORMAP_JET 伪彩色映射规则
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # BGR转RGB以供Streamlit正常显示
            
            # 多模态线性加权叠加融合 (原始胸片权重 0.6，伪彩色热力图权重 0.4)
            blend_view = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
            
            # 5. 联动展现：Grad-CAM 视觉循证区挂载
            st.write("#### 🔍 Grad-CAM 病灶临床可解释性定位")
            st.image(blend_view, caption="高饱和度红色区域标定为算法核心决策靶区（炎性渗出密集区）", use_container_width=True)
            
            # 6. 状态4：闭环导出自动结构化电子辅助诊断报告
            st.write("#### 📋 自动生成的电子辅助诊断报告")
            if pneumonia_risk > 50:
                diagnostic_opinion = "【阳性，提示存在明显的肺炎炎性浸润影】"
                status_color = st.error
            else:
                diagnostic_opinion = "【阴性，未见明显肺泡实变及炎性渗出阴影】"
                status_color = st.success
                
            status_color(f"""
            **放射科 AI 辅助筛查意见书**
            * 📄 **患者档案信息**: {p_id} ｜ {p_name} ｜ {p_gender} ｜ {p_age} 岁
            * 🎯 **AI 筛查量化结论**: {diagnostic_opinion} (肺炎置信度: {pneumonia_risk:.2f}%)
            * 🧭 **空间循证提示**: 算法模型决策激活区主要聚焦于图像高亮热力覆盖段，提示有斑片状高密度影或磨玻璃样改变。
            * ⚠️ **免责声明**: 本电子报告仅作为放射科医生初筛质控的数字化线索辅助，最终确诊结果请以资深临床执业医师的会诊意见为准。
            """)
        else:
            st.info("💡 请在左侧上传胸片，AI解算中心将在此实时渲染联动分析视图。")

if __name__ == "__main__":
    main()

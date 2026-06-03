# -*- coding: utf-8 -*-
"""
DenseNet-121 肺炎分类器训练与微调流线
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os

def main():
    # ==========================================
    # 1. 超参数与配置中心
    # ==========================================
    DATA_DIR = "./dataset"  # 👈 填写你数据集的根目录路径
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4    # 迁移学习微调通常使用较小的学习率
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前训练计算舱挂载设备: {DEVICE}")

    # ==========================================
    # 2. 医疗影像专属数据增强与流线预处理
    # ==========================================
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),      # 增加模型左右肺泛化鲁棒性
            transforms.RandomRotation(15),          # 模拟医生摄片时的轻微体位倾斜
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 匹配ImageNet标准
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2) for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"📊 数据集初始化完成: 训练集 {dataset_sizes['train']} 张, 验证集 {dataset_sizes['val']} 张 | 分类标签: {class_names}")

    # ==========================================
    # 3. 骨干网络拓扑构建 (重塑分类舱)
    # ==========================================
    # 加载预训练的 DenseNet-121
    model = models.densenet121(pretrained=True)
    
    # 替换最后的全连接层，使其适配我们的二分类（Normal vs Pneumonia）
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)

    # 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 4. 标准前向-反向迭代训练循环
    # ==========================================
    best_acc = 0.0
    best_model_wts = model.state_dict()

    print("\n🚀 神经网络核心训练引擎启动...")
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 20)

        # 每个Epoch包含训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 激活训练模式（允许梯度计算、Dropout和BatchNorm更新）
            else:
                model.eval()   # 激活关闭模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历批次数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 梯度清零
                optimizer.zero_grad()

                # 正向传播 (仅在训练阶段追踪梯度)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 逆向求解与参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计量化指标
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度质控：如果发现更优权重，执行内存常驻备份
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print(f"🌟 检测到更优的验证集准确率: {best_acc:.4f}，已暂存当前模型权重。")

    print(f'\n🎯 整个训练流线结束！最优验证集准确率 (Best Val Acc): {best_acc:4f}')

    # ==========================================
    # 5. 权重固化与本地落盘
    # ==========================================
    model.load_state_dict(best_model_wts)
    output_weight_name = "best_model.pth"
    torch.save(model.state_dict(), output_weight_name)
    print(f"💾 核心权重已成功落盘至本地: {os.path.abspath(output_weight_name)}")

if __name__ == '__main__':
    main()

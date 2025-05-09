import os
import torch
import pandas as pd
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F

def finetune_model(model, train_dataloader, epochs=1, device='cuda'):
    """剪枝后的模型微调"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) if 'labels' in batch else None

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, eval_dataloader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) if 'labels' in batch else None

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            if labels is not None:
                total_loss += outputs.loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    return avg_loss


def get_model_size(model):
    """计算模型大小（MB）"""
    torch.save(model.state_dict(), "temp_model.pt")
    size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
    os.remove("temp_model.pt")
    return size_mb

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker


def plt_snr_performance(results=None, channel_type=None):
    # 设置绘图风格
    if results is None:
        results = {}
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })

    # 创建高分辨率图形
    plt.figure(figsize=(14, 6), dpi=150)

    # 定义颜色和标记样式
    colors = ['#2878B5', '#9AC9DB', '#F8AC8C', '#C82423', '#FF8884']

    # SNR vs. 损失
    ax1 = plt.subplot(1, 2, 1)
    snrs = list(results.keys())
    losses = [results[snr]['loss'] for snr in snrs]

    # 平滑曲线效果 (可选)
    ax1.plot(snrs, losses, '-', color=colors[0], linewidth=2.5, alpha=0.7)
    ax1.plot(snrs, losses, 'o', color=colors[0], markersize=8,
            markeredgecolor='white', markeredgewidth=1.5)

    # 设置轴格式
    ax1.set_xlabel('SNR (dB)', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Loss vs. SNR', fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 优化轴刻度
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(direction='out', length=6, width=1.5, colors='black')
    if len(snrs) <= 10:  # 如果数据点较少，显示所有刻度
        ax1.set_xticks(snrs)

    # 添加数据值标签 (可选)
    for i, loss in enumerate(losses):
        ax1.annotate(f'{loss:.4f}',
                    xy=(snrs[i], loss),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # SNR vs. BLEU
    ax2 = plt.subplot(1, 2, 2)
    bleus = [results[snr]['bleu'] for snr in snrs]

    # 绘制阴影区域表示趋势
    ax2.fill_between(snrs,
                    [max(0, b-0.005) for b in bleus],
                    [min(1, b+0.005) for b in bleus],
                    alpha=0.2, color=colors[3])

    # 绘制主曲线
    ax2.plot(snrs, bleus, '-', color=colors[3], linewidth=2.5, alpha=0.7)
    ax2.plot(snrs, bleus, 'o', color=colors[3], markersize=8,
            markeredgecolor='white', markeredgewidth=1.5)

    # 设置轴格式
    ax2.set_xlabel('SNR (dB)', fontweight='bold')
    ax2.set_ylabel('BLEU Score', fontweight='bold')
    ax2.set_title('BLEU Score vs. SNR', fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 优化轴刻度
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(direction='out', length=6, width=1.5, colors='black')
    if len(snrs) <= 10:  # 如果数据点较少，显示所有刻度
        ax2.set_xticks(snrs)

    # 添加数据值标签 (可选)
    for i, bleu in enumerate(bleus):
        ax2.annotate(f'{bleu:.4f}',
                    xy=(snrs[i], bleu),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # 添加整体标题
    plt.suptitle('Performance Metrics vs. Signal-to-Noise Ratio',
                fontsize=16, fontweight='bold', y=0.98)

    # 确保布局紧凑且没有重叠
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # rect参数为[left, bottom, right, top]

    # 保存高分辨率图片
    plt.savefig(f'{channel_type}_snr_performance_enhanced3.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_metrics_to_csv(train_losses, valid_losses, bleu_scores, filepath, channel_type=None):
    """
    将训练指标保存到CSV文件

    参数:
    train_losses (list): 训练损失列表
    valid_losses (list): 验证损失列表
    bleu_scores (list): BLEU评分列表
    filepath (str): 保存CSV文件的路径
    """
    # 确保所有列表长度一致，如果不一致则用None填充
    max_length = max(len(train_losses), len(valid_losses), len(bleu_scores))

    train_losses = train_losses + [None] * (max_length - len(train_losses))
    valid_losses = valid_losses + [None] * (max_length - len(valid_losses))
    bleu_scores = bleu_scores + [None] * (max_length - len(bleu_scores))

    # 创建数据框
    data = {
        'epoch': list(range(1, max_length + 1)),
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'bleu_score': bleu_scores
    }

    df = pd.DataFrame(data)

    # 确保目录存在
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    # 保存到CSV
    df.to_csv(filepath, index=False)
    print(f"训练指标已保存到 {filepath}")



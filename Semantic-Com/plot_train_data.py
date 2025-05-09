import matplotlib
# 尝试不同的后端，例如 'TkAgg' 或 'Qt5Agg'
# 如果在一个后端出错，可以注释掉这一行，尝试另一个
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_distillation_results(csv_file):
    """
    读取CSV文件并绘制知识蒸馏实验结果图表
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file)

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    ax1.plot(data['Epoch'], data['Train_Loss'], 'b-', label='Train Loss')
    ax1.plot(data['Epoch'], data['Valid_Loss'], 'orange', label='Valid Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    # ax1.grid(True, linestyle='--', alpha=0.7)
    # ax1.set_ylim(0, max(data['Train_Loss'].max(), data['Valid_Loss'].max()) * 1.1)

    # 绘制BLEU分数曲线
    ax2.plot(data['Epoch'], data['BLEU_Score'], 'b-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BLEU Score')
    ax2.set_title('BLEU Score on Validation Set')
    # ax2.grid(True, linestyle='--', alpha=0.7)
    # ax2.set_ylim(0, 100)

    # 调整布局并保存图片
    # plt.tight_layout()
    # plt.savefig('distillation_results.png', dpi=300, bbox_inches='tight')
    # plt.show()
    #
    # print(f"图表已保存为 'distillation_results.png'")

    plt.tight_layout()
    plt.savefig('final_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"图表已保存为 'final_results.png'")


# 调用函数绘制图表
# plot_distillation_results('distil_train.csv')
plot_distillation_results('final_train.csv')
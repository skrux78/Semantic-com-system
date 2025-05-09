import matplotlib
# 尝试不同的后端，例如 'TkAgg' 或 'Qt5Agg'
# 如果在一个后端出错，可以注释掉这一行，尝试另一个
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # 用于设置坐标轴刻度格式

# 设置 matplotlib 风格以获得更美观的图表
plt.style.use('seaborn-v0_8-whitegrid') # 使用 seaborn 的白网格风格作为基础

# 读取 CSV 文件
# csv_filename = 'BERT_performance_vs_snr.csv'
# csv_filename = 'distil_performance_vs_snr.csv'
csv_filename = 'final_performance_vs_snr.csv'
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"错误：找不到文件 {csv_filename}。请先运行 save_performance_data.py 生成数据文件。")
    exit()

# 获取 SNR 值
snr_values = df['SNR']

# 识别 Loss 和 BLEU 指标列
# 假定列名以 '_Loss' 结尾的是 Loss 指标
# 假定列名以 '_BLEU' 结尾的是 BLEU 指标
loss_cols = [col for col in df.columns if col.endswith('_Loss')]
bleu_cols = [col for col in df.columns if col.endswith('_BLEU')]

# 提取模型名称 (用于图例标签)
# 例如 'Base_BLEU' 提取出 'Base'
def get_model_label(col_name):
    if col_name.endswith('_Loss'):
        return col_name.replace('_Loss', '')
    elif col_name.endswith('_BLEU'):
        return col_name.replace('_BLEU', '')
    return col_name # 如果不符合命名规则，直接使用列名

# 创建一个包含两个子图的 Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # 1行2列，设置 Figure 大小

# 绘制 Loss vs. SNR 图 (左侧子图)
ax1 = axes[0]
for col in loss_cols:
    model_label = get_model_label(col)
    ax1.plot(snr_values, df[col], marker='o', label=model_label) # 绘制曲线和标记
    # 添加数据标签 (注释)
    for x, y in zip(snr_values, df[col]):
        # 格式化 Loss 值，保留更多小数位
        ax1.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8))

ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss vs. SNR', fontsize=14)
ax1.legend()
# 调整网格样式以匹配图片
ax1.grid(True, linestyle='--', alpha=0.6)
# 移除顶部和右侧的边框
ax1.spines[['right', 'top']].set_visible(False)

# 设置 Loss 轴的刻度格式，保留特定小数位数 (根据您的数据调整)
# 例如，强制显示4位小数
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))


# 绘制 BLEU Score vs. SNR 图 (右侧子图)
ax2 = axes[1]
# 遍历 BLEU 列并绘制
for i, col in enumerate(bleu_cols):
    model_label = get_model_label(col)
    line, = ax2.plot(snr_values, df[col], marker='o', label=model_label) # 绘制曲线和标记
    # 添加数据标签 (注释)
    for x, y in zip(snr_values, df[col]):
         # 格式化 BLEU 值，保留2位小数位 (或您需要的精度)
        ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8))

    # 为第一条 BLEU 曲线添加填充区域 (可选，可以根据需要删除或修改)
    if i == 0: # 例如，只为基准模型的 BLEU 曲线填充
        ax2.fill_between(snr_values, 0, df[col], color=line.get_color(), alpha=0.1, label=f'{model_label} Area') # 填充到 y=0

ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('BLEU Score', fontsize=12)
ax2.set_title('BLEU Score vs. SNR', fontsize=14)
ax2.legend()
# 调整网格样式以匹配图片
ax2.grid(True, linestyle='--', alpha=0.6)
# 移除顶部和右侧的边框
ax2.spines[['right', 'top']].set_visible(False)

# 设置 BLEU Score 轴的刻度格式，保留特定小数位数
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# 设置整个 Figure 的大标题
fig.suptitle('Performance Metrics vs. Signal-to-Noise Ratio', fontsize=16, y=1.02) # y 调整标题位置

# 调整子图之间的间距
plt.tight_layout()

# 显示图表
plt.show()

# 可以选择保存图表为文件
# plt.savefig('distil_performance_vs_snr.png', dpi=300, bbox_inches='tight')
plt.savefig('final_performance_vs_snr.png', dpi=300, bbox_inches='tight')
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from BERT_base_train import SemanticCommunicationSystem, TEST_SNRS, prepare_data
from BERT_base_train import validate, Channel_type, plt_snr_performance
import torch.quantization
import time
import matplotlib as mpl
from tqdm import tqdm


# 解决中文字符显示问题
def configure_matplotlib_fonts():
    """配置matplotlib以支持中文显示"""
    try:
        # 方法1：使用系统中的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei'] + \
                                          plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 测试中文显示
        fig = plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
        plt.close(fig)

        print("成功配置中文字体")
    except Exception as e:
        print(f"配置中文字体时出错: {str(e)}")
        print("将使用英文标签替代中文")
        # 如果无法正确显示中文，则使用英文标签
        global use_english_labels
        use_english_labels = True

    # 默认使用中文标签，如果配置失败则切换到英文


use_english_labels = False


def get_label(cn_text, en_text):
    """根据字体配置返回中文或英文标签"""
    return en_text if use_english_labels else cn_text

def benchmark_inference_speed(model, device, dataloader, num_runs=50):
    model.eval()
    batch = next(iter(dataloader))

    # 移动数据到设备
    src_input_ids = batch['src_input_ids'].to(device)
    src_attention_mask = batch['src_attention_mask'].to(device)
    tgt_input_ids = batch['tgt_input_ids'].to(device)
    tgt_attention_mask = batch['tgt_attention_mask'].to(device)

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, channel=Channel_type)

            # 计时
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, channel=Channel_type)

    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_runs
    return avg_time

def calculate_flops(model, device, data_loader, channel_type=None):
    """使用实际训练样例计算模型的FLOPs"""
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        import copy

        # 获取一个真实样例
        data_iter = iter(data_loader)
        batch = next(data_iter)

        # 打印批次中的所有键值对
        print("\n===== 批次数据结构 =====")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: 形状{value.shape}, 类型{value.dtype}")
                # 打印该张量的一小部分示例值
                if value.numel() > 0:
                    flat_value = value.view(-1)
                    print(f"  示例值: {flat_value[:5].tolist()}...")
            else:
                print(f"{key}: {type(value)}")

                # 准备输入（与train_epoch中相同）
        src_input_ids = batch['src_input_ids'].to(device)
        src_attention_mask = batch['src_attention_mask'].to(device)
        tgt_input_ids = batch['tgt_input_ids'].to(device)
        tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        # 打印第一个样例的实际内容
        print("\n===== 第一个样例详情 =====")
        print(f"源输入ID: {src_input_ids[0][:20].tolist()}...")  # 显示前20个token
        print(f"源掩码: {src_attention_mask[0][:20].tolist()}...")
        print(f"目标输入ID: {tgt_input_ids[0][:20].tolist()}...")
        print(f"目标掩码: {tgt_attention_mask[0][:20].tolist()}...")

        # 打印样例尺寸信息
        print(f"\n===== 样例尺寸 =====")
        print(f"源序列长度: {src_input_ids.shape}")
        print(f"目标序列长度: {tgt_input_ids.shape}")

        # 创建模型的副本用于FLOPs计算
        model_copy = copy.deepcopy(model)
        model_copy.eval()

        # 准备输入（与train_epoch中相同）
        src_input_ids = batch['src_input_ids'].to(device)
        src_attention_mask = batch['src_attention_mask'].to(device)
        tgt_input_ids = batch['tgt_input_ids'].to(device)
        tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        # 打印样例尺寸信息
        print(f"样例尺寸：")
        print(f"源序列长度: {src_input_ids.shape}")
        print(f"目标序列长度: {tgt_input_ids.shape}")

        # 设定输入参数
        inputs = (src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask)
        if channel_type is not None:
            # 如果计算带有channel_type的FLOPs，需要准备不同的模型调用
            # FLOP分析会执行无参数的forward函数，所以我们需要定义一个包装器
            def wrapper(model, inputs):
                src_ids, src_mask, tgt_ids, tgt_mask = inputs
                return model(src_ids, src_mask, tgt_ids, tgt_mask, channel=channel_type)

                # 直接传递模型和输入给FlopCountAnalysis

            flops_analysis = FlopCountAnalysis(lambda x: wrapper(model_copy, x), inputs)
        else:
            # 如果不需要channel_type，直接使用原始inputs
            flops_analysis = FlopCountAnalysis(model_copy, inputs)

            # 禁用警告
        flops_analysis.unsupported_ops_warnings(False)
        flops_analysis.uncalled_modules_warnings(False)

        # 获取总FLOPs
        total_flops = flops_analysis.total()

        # 打印详细的FLOPs表
        print(flop_count_table(flops_analysis))

        if total_flops > 1e9:
            print(f"模型FLOPs: {total_flops / 1e9:.4f} GFLOPs")
        else:
            print(f"模型FLOPs: {total_flops / 1e6:.4f} MFLOPs")

        return total_flops

    except ImportError:
        print("请安装fvcore以计算FLOPs: pip install fvcore")
        return 0
    except Exception as e:
        print(f"计算FLOPs时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0


def count_parameters(model):
    """计算模型的参数数量（百万）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def simple_flops_estimation(model, input_shape=(1, 128)):
    """简单估计模型的FLOPs（基于参数数量）"""
    # 这是一个粗略估计，实际FLOPs会受到模型架构的影响
    num_params = sum(p.numel() for p in model.parameters())

    # 假设每个参数平均参与2次计算
    estimated_flops = num_params * 2

    # 考虑序列长度的影响
    seq_length = input_shape[1]
    estimated_flops *= seq_length

    if estimated_flops > 1e9:
        print(f"估计的FLOPs: {estimated_flops / 1e9:.4f} GFLOPs")
    else:
        print(f"估计的FLOPs: {estimated_flops / 1e6:.4f} MFLOPs")

    return estimated_flops


def test_at_different_snrs(model, test_loader, tokenizer):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cpu_device = torch.device("cpu")
    cpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== 加载原始模型 ===")
    # 加载原始模型
    original_model = SemanticCommunicationSystem().to(device)
    original_model.load_state_dict(torch.load('test_models/best_model.pt', weights_only=True))
    original_model.eval()

    print("\n=== 加载量化模型 ===")
    # 加载量化模型
    # quantized_device = torch.device("cpu")  # 量化模型只能在CPU上运行
    quantized_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        quantized_model = torch.load("quantized_full_model.pt")
        quantized_model = quantized_model.to(quantized_device)
        quantized_model.eval()
    except Exception as e:
        print(f"加载量化模型时出错: {str(e)}")
        return

        # 计算FLOPs - 使用简单估计方法作为备选
    print("\n=== 计算原始模型FLOPs ===")
    try:
        original_flops = calculate_flops(original_model, device,test_loader)
        if original_flops == 0:
            print("使用简单方法估计FLOPs...")
            original_flops = simple_flops_estimation(original_model)
    except Exception as e:
        print(f"计算原始模型FLOPs时发生错误: {str(e)}")
        original_flops = simple_flops_estimation(original_model)

    print("\n=== 计算量化模型FLOPs ===")
    try:
        quantized_flops = calculate_flops(quantized_model, quantized_device,test_loader)
        if quantized_flops == 0:
            print("使用简单方法估计FLOPs...")
            quantized_flops = simple_flops_estimation(quantized_model)
    except Exception as e:
        print(f"计算量化模型FLOPs时发生错误: {str(e)}")
        quantized_flops = simple_flops_estimation(quantized_model)

        # 显示FLOPs对比
    if original_flops and quantized_flops:
        flops_reduction = (original_flops - quantized_flops) / original_flops * 100
        print("\n=== FLOPs对比 ===")
        print(f"FLOPs减少: {flops_reduction:.2f}%")

    print("\n=== 计算原始模型推理速度 ===")
    print(benchmark_inference_speed(original_model, device, test_loader))

    print("\n=== 计算量化模型推理速度 ===")
    print(benchmark_inference_speed(quantized_model, quantized_device, test_loader))


        # 存储不同SNR下的性能
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 测试量化模型 - 使用更安全的方式
    print("\n=== 测试量化模型在不同SNR下的性能 ===")
    quantized_results = {}

    # 只测试部分SNR以节省时间
    test_snrs = [0, 10, 20]  # 可以改为 [0, 10, 20] 等少量值以加快测试

    for snr_db in test_snrs:
        try:
            print(f"测试量化模型在SNR={snr_db}dB...")
            valid_loss, bleu_score = validate(quantized_model, test_loader, tokenizer, criterion, snr_db=snr_db)
            quantized_results[snr_db] = {
                'loss': valid_loss,
                'bleu': bleu_score
            }
            print(f"量化模型 - SNR: {snr_db}dB - Loss: {valid_loss:.4f}, BLEU: {bleu_score:.2f}")
        except Exception as e:
            print(f"在SNR={snr_db}dB测试量化模型时出错: {str(e)}")
            quantized_results[snr_db] = {
                'loss': float('nan'),
                'bleu': 0.0
            }

            # 测试原始模型
    print("\n=== 测试原始模型在不同SNR下的性能 ===")
    original_results = {}
    for snr_db in test_snrs:
        try:
            print(f"测试原始模型在SNR={snr_db}dB...")
            valid_loss, bleu_score = validate(original_model, test_loader, tokenizer, criterion, snr_db=snr_db)
            original_results[snr_db] = {
                'loss': valid_loss,
                'bleu': bleu_score
            }
            print(f"原始模型 - SNR: {snr_db}dB - Loss: {valid_loss:.4f}, BLEU: {bleu_score:.2f}")
        except Exception as e:
            print(f"在SNR={snr_db}dB测试原始模型时出错: {str(e)}")
            original_results[snr_db] = {
                'loss': float('nan'),
                'bleu': 0.0
            }

            # 保存结果到CSV
    save_results_to_csv(original_results, quantized_results, original_flops, quantized_flops)

    # 绘制图表
    try:
        plt_snr_performance(quantized_results, channel_type=Channel_type, label="Quantized Model")
        plt_snr_performance(original_results, channel_type=Channel_type, label="Original Model")
    except Exception as e:
        print(f"绘制SNR性能图表时出错: {str(e)}")

        # 绘制FLOPs对比图
    try:
        plot_flops_comparison(original_flops, quantized_flops, original_results, quantized_results)
    except Exception as e:
        print(f"绘制FLOPs对比图时出错: {str(e)}")

    return original_results, quantized_results, original_flops, quantized_flops


def save_results_to_csv(original_results, quantized_results, original_flops, quantized_flops):
    """保存对比结果到CSV文件"""
    # 确保目录存在
    os.makedirs('metrics', exist_ok=True)

    # 创建SNR性能数据
    data = {
        'SNR': list(original_results.keys()),
        'Original_Loss': [original_results[snr]['loss'] for snr in original_results],
        'Original_BLEU': [original_results[snr]['bleu'] for snr in original_results],
        'Quantized_Loss': [quantized_results[snr]['loss'] for snr in quantized_results],
        'Quantized_BLEU': [quantized_results[snr]['bleu'] for snr in quantized_results]
    }

    df = pd.DataFrame(data)
    df.to_csv('metrics/model_comparison.csv', index=False)

    # 创建FLOPs数据
    flops_data = {
        'Model': ['Original Model', 'Quantized Model'],
        'FLOPs': [original_flops, quantized_flops],
        'FLOPs_GFLOPs': [original_flops / 1e9, quantized_flops / 1e9],
        'FLOPs_MFLOPs': [original_flops / 1e6, quantized_flops / 1e6],
        'FLOPs_Reduction_Percent': [0, (original_flops - quantized_flops) / original_flops * 100]
    }

    flops_df = pd.DataFrame(flops_data)
    flops_df.to_csv('metrics/flops_comparison.csv', index=False)

    print(f"对比结果已保存到 metrics/model_comparison.csv")
    print(f"FLOPs对比已保存到 metrics/flops_comparison.csv")


def plot_flops_comparison(original_flops, quantized_flops, original_results, quantized_results):
    """绘制FLOPs对比图和性能效率图"""
    plt.figure(figsize=(15, 10))

    # 1. FLOPs对比柱状图
    plt.subplot(2, 2, 1)
    labels = [get_label('原始模型', 'Original Model'), get_label('量化模型', 'Quantized Model')]
    if original_flops > 1e9:
        flops_values = [original_flops / 1e9, quantized_flops / 1e9]
        flops_unit = 'GFLOPs'
    else:
        flops_values = [original_flops / 1e6, quantized_flops / 1e6]
        flops_unit = 'MFLOPs'

    bars = plt.bar(labels, flops_values, color=['blue', 'orange'])
    plt.ylabel(flops_unit)
    plt.title(get_label('计算量(FLOPs)对比', 'FLOPs Comparison'))

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

        # 添加减少百分比标注
    reduction = (original_flops - quantized_flops) / original_flops * 100
    plt.text(0.5, 0.5, f"Reduction: {reduction:.2f}%",
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # 2. BLEU vs. SNR对比
    plt.subplot(2, 2, 2)
    snrs = list(original_results.keys())
    plt.plot(snrs, [original_results[snr]['bleu'] for snr in snrs], 'o-', label='Original Model')
    plt.plot(snrs, [quantized_results[snr]['bleu'] for snr in snrs], 's-', label='Quantized Model')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score vs. SNR')
    plt.grid(True)
    plt.legend()

    # 3. 计算效率图(BLEU/GFLOPs)
    plt.subplot(2, 2, 3)
    # 计算每个SNR下的平均BLEU
    original_avg_bleu = sum(original_results[snr]['bleu'] for snr in snrs) / len(snrs)
    quantized_avg_bleu = sum(quantized_results[snr]['bleu'] for snr in snrs) / len(snrs)

    # 计算效率(BLEU/GFLOPs)
    original_efficiency = original_avg_bleu / (original_flops / 1e9)
    quantized_efficiency = quantized_avg_bleu / (quantized_flops / 1e9)

    bars = plt.bar([get_label('原始模型', 'Original Model'), get_label('量化模型', 'Quantized Model')],
                   [original_efficiency, quantized_efficiency],
                   color=['blue', 'orange'])
    plt.ylabel('Efficiency (BLEU/GFLOPs)')
    plt.title(get_label('计算效率对比 (BLEU/GFLOPs)', 'Computational Efficiency (BLEU/GFLOPs)'))

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

        # 4. BLEU vs FLOPs散点图
    plt.subplot(2, 2, 4)
    plt.scatter([original_flops / 1e9], [original_avg_bleu], s=100, c='blue', label='Original Model')
    plt.scatter([quantized_flops / 1e9], [quantized_avg_bleu], s=100, c='orange', label='Quantized Model')

    # 添加标签
    plt.annotate(f'Original: {original_avg_bleu:.2f}',
                 (original_flops / 1e9, original_avg_bleu),
                 xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'Quantized: {quantized_avg_bleu:.2f}',
                 (quantized_flops / 1e9, quantized_avg_bleu),
                 xytext=(10, 10), textcoords='offset points')

    # 连接两个点
    plt.plot([original_flops / 1e9, quantized_flops / 1e9],
             [original_avg_bleu, quantized_avg_bleu],
             'k--', alpha=0.5)

    plt.xlabel('Computation (GFLOPs)')
    plt.ylabel('Average BLEU Score')
    plt.title(get_label('性能-计算量权衡', 'Performance-Computation Trade-off'))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics/flops_performance_comparison.png')
    print(f"FLOPs性能对比图已保存到 metrics/flops_performance_comparison.png")


def model_size_info(model, name="Model"):
    """计算模型大小和参数数量"""
    # 计算参数数量
    num_params = sum(p.numel() for p in model.parameters())

    # 计算模型大小(MB)
    model_size = 0
    for param in model.parameters():
        model_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        model_size += buffer.nelement() * buffer.element_size()

    model_size_mb = model_size / (1024 * 1024)

    print(f"{name} parameters: {num_params:,}")
    print(f"{name} size: {model_size_mb:.2f} MB")

    return model_size_mb, num_params


def compare_model_sizes():
    """比较原始模型和量化模型的大小"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    # 加载原始模型
    original_model = SemanticCommunicationSystem().to(device)
    original_model.load_state_dict(torch.load('test_models/best_model.pt', weights_only=True))

    # 计算原始模型大小
    print("\n=== Original Model Information ===")
    orig_size_mb, orig_params = model_size_info(original_model, "Original model")

    # 加载量化模型
    try:
        quantized_model = torch.load("quantized_full_model.pt")
        quantized_model = quantized_model.to(cpu_device)

        # 计算量化模型大小
        print("\n=== Quantized Model Information ===")
        quant_size_mb, quant_params = model_size_info(quantized_model, "Quantized model")

        # 计算减少比例
        size_reduction = (orig_size_mb - quant_size_mb) / orig_size_mb * 100
        params_reduction = (orig_params - quant_params) / orig_params * 100

        print(f"\nModel size reduction: {size_reduction:.2f}%")
        print(f"Parameter count reduction: {params_reduction:.2f}%")

        # 创建对比图
        plt.figure(figsize=(12, 5))

        # 模型大小对比
        plt.subplot(1, 2, 1)
        bars = plt.bar([get_label('原始模型', 'Original Model'), get_label('量化模型', 'Quantized Model')],
                       [orig_size_mb, quant_size_mb])
        plt.ylabel('Model Size (MB)')
        plt.title('Model Size Comparison')

        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')

            # 参数数量对比
        plt.subplot(1, 2, 2)
        bars = plt.bar([get_label('原始模型', 'Original Model'), get_label('量化模型', 'Quantized Model')],
                       [orig_params / 1e6, quant_params / 1e6])
        plt.ylabel('Parameter Count (millions)')
        plt.title('Parameter Count Comparison')

        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('metrics/model_size_comparison.png')
        print(f"模型大小对比图已保存到 metrics/model_size_comparison.png")

        return orig_size_mb, quant_size_mb, orig_params, quant_params

    except Exception as e:
        print(f"加载量化模型时出错: {str(e)}")
        return orig_size_mb, None, orig_params, None


def benchmark_inference_speed(model, device, dataloader, num_runs=50):
    model.eval()
    batch = next(iter(dataloader))

    # 移动数据到设备
    src_input_ids = batch['src_input_ids'].to(device)
    src_attention_mask = batch['src_attention_mask'].to(device)
    tgt_input_ids = batch['tgt_input_ids'].to(device)
    tgt_attention_mask = batch['tgt_attention_mask'].to(device)

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, channel=Channel_type)

            # 计时
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, channel=Channel_type)

    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_runs
    return avg_time


def main():
    """主函数"""
    try:
        # 配置matplotlib以支持中文
        configure_matplotlib_fonts()

        # 确保metrics目录存在
        os.makedirs('metrics', exist_ok=True)

        # 加载数据
        _, _, test_loader, _, tokenizer = prepare_data()

        # 比较模型大小
        compare_model_sizes()

        # 测试模型在不同SNR下的性能并比较FLOPs
        # 为了简洁起见，这里传入None作为model参数，在函数内部加载模型
        test_at_different_snrs(None, test_loader, tokenizer)

        print("\n===== 分析完成 =====")

    except Exception as e:
        print(f"主函数执行时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
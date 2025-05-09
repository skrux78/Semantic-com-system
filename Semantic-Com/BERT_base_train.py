import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import font_manager
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, MarianMTModel, MarianTokenizer
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from sacrebleu.metrics import BLEU
import matplotlib.pyplot as plt
from tqdm import tqdm
# from torch.cuda.amp import autocast, GradScaler  # 添加混合精度训练支持
from torch.amp import autocast, GradScaler
from util import plt_snr_performance, save_metrics_to_csv

def inspect_dataset(dataset):
    """检查并打印数据集的结构"""
    print("\n数据集检查:")
    print(f"数据集类型: {type(dataset)}")
    print(f"数据集键: {list(dataset.keys())}")

    # 检查训练集的第一个样本
    sample = dataset['train'][0]
    print(f"\n样本类型: {type(sample)}")
    print(f"样本键: {list(sample.keys())}")
    print(f"样本内容: {sample}")

    return sample


# 设置随机种子以确保可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# 测试量化
device = torch.device("cpu")

# 配置参数 - 优化后
MAX_LENGTH = 64  # 减少句子最大长度 (从128降到64) 完整用64可以收敛 64 128
BATCH_SIZE = 32  # 减小批次大小以降低内存使用 16 16
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数，有效批次大小 = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 2e-5
EPOCHS = 50
SEMANTIC_DIM = 128  # 减少语义特征维度 (从256降到128) 128 128
Channel_type = 'awgn'  #  'awgn': AWGNChannel(),'rayleigh': RayleighChannel(),'rician': RicianChannel()
TRAINING_SNR = 12  # 训练时的信噪比(dB)
TEST_SNRS = [0, 3, 6, 9, 12, 15, 18]  # 测试时的信噪比(dB)列表
DEV_MODE = False  # 开发模式标志，设为True时使用较小的数据集加快开发 False True
DEV_SUBSET_SIZE = 5000  # 开发模式下的训练集样本数量 0 100000


# --------------------- 1. 数据预处理 ---------------------

class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        # 确定数据集的结构（在初始化时检查第一个样本）
        sample = data[0]
        self.has_translation_key = 'translation' in sample
        self.has_en_de_keys = 'en' in sample and 'de' in sample

        if not self.has_translation_key and not self.has_en_de_keys:
            # 尝试打印样本以帮助诊断
            print(f"警告：无法识别的数据集结构: {sample.keys()}")
            print(f"样本内容: {sample}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 根据不同的数据集结构提取源语言和目标语言文本
        if self.has_translation_key:
            # 如果数据集使用'translation'键
            src_text = self.data[idx]['translation']['en']
            tgt_text = self.data[idx]['translation']['de']
        elif self.has_en_de_keys:
            # 如果数据集直接使用'en'和'de'键
            src_text = self.data[idx]['en']
            tgt_text = self.data[idx]['de']
        else:
            # 尝试其他可能的键名组合
            # IWSLT2017数据集可能使用'sourceString'和'targetString'
            keys = list(self.data[idx].keys())
            if 'sourceString' in keys and 'targetString' in keys:
                src_text = self.data[idx]['sourceString']
                tgt_text = self.data[idx]['targetString']
            else:
                # 最后的尝试 - 假设第一个键是源语言，第二个键是目标语言
                src_text = self.data[idx][keys[0]]
                tgt_text = self.data[idx][keys[1]]

        src_encoding = self.src_tokenizer(
            src_text,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True
        )

        tgt_encoding = self.tgt_tokenizer(
            tgt_text,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True
        )

        return {
            'src_input_ids': src_encoding['input_ids'].squeeze(),
            'src_attention_mask': src_encoding['attention_mask'].squeeze(),
            'tgt_input_ids': tgt_encoding['input_ids'].squeeze(),
            'tgt_attention_mask': tgt_encoding['attention_mask'].squeeze(),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def prepare_data():
    print("Loading IWSLT'17 En-De dataset...")

    # 加载IWSLT数据集
    dataset = load_dataset("iwslt2017", "iwslt2017-en-de", trust_remote_code=True)

    # 如果开发模式打开，使用数据子集加快开发
    if DEV_MODE:
        print(f"开发模式: 使用训练集的{DEV_SUBSET_SIZE}个样本")
        dataset['train'] = dataset['train'].select(range(min(DEV_SUBSET_SIZE, len(dataset['train']))))
        dataset['validation'] = dataset['validation'].select(range(min(1000, len(dataset['validation']))))
        dataset['test'] = dataset['test'].select(range(min(1000, len(dataset['test']))))

        # 初始化分词器
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

    # 数据集检查
    sample = inspect_dataset(dataset)

    # 创建数据集
    train_data = TranslationDataset(dataset['train'], src_tokenizer, tgt_tokenizer)
    valid_data = TranslationDataset(dataset['validation'], src_tokenizer, tgt_tokenizer)
    test_data = TranslationDataset(dataset['test'], src_tokenizer, tgt_tokenizer)

    # 创建数据加载器 - 使用num_workers和pin_memory优化
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # 使用多进程加载数据
        pin_memory=True  # 将数据直接加载到GPU内存
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    print(f"Dataset sizes - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    return train_loader, valid_loader, test_loader, src_tokenizer, tgt_tokenizer


# --------------------- 2. 模型定义 ---------------------

class SemanticEncoder(nn.Module):
    def __init__(self):
        super(SemanticEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.feature_projection = nn.Linear(768, SEMANTIC_DIM)

    def forward(self, input_ids, attention_mask):
        # 使用BERT提取特征
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 使用[CLS]标记的表示作为句子语义特征
        cls_embedding = last_hidden_state[:, 0]

        # 投影到语义特征维度
        semantic_features = self.feature_projection(cls_embedding)
        return semantic_features


class ChannelEncoder(nn.Module):
    def __init__(self, semantic_dim=SEMANTIC_DIM, channel_dim=SEMANTIC_DIM):
        super(ChannelEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(semantic_dim, channel_dim),
            nn.Tanh()  # 限制输出范围，使信号能量归一化
        )

    def forward(self, x):
        return self.encoder(x)


class ChannelDecoder(nn.Module):
    def __init__(self, channel_dim=SEMANTIC_DIM, semantic_dim=SEMANTIC_DIM):
        super(ChannelDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(channel_dim, semantic_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)


class Channel(nn.Module):
    """信道模型基类"""

    def __init__(self):
        super(Channel, self).__init__()

    def forward(self, x, snr_db, **kwargs):
        """
        信道模型的前向传播

        Args:
            x: 输入信号
            snr_db: 信噪比(dB)
            **kwargs: 额外的信道参数

        Returns:
            接收信号
        """
        raise NotImplementedError("子类必须实现forward方法")


class AWGNChannel(Channel):
    # def add_noise(self, x, snr_db):
    #     """AWGN信道模型"""
    #     # 转换SNR从dB到线性尺度
    #     snr_linear = 10 ** (snr_db / 10)
    #
    #     # 计算信号功率
    #     signal_power = torch.mean(torch.abs(x) ** 2)
    #
    #     # 计算噪声功率
    #     noise_power = signal_power / snr_linear
    #
    #     # 生成高斯噪声
    #     noise = torch.sqrt(noise_power) * torch.randn_like(x)
    #
    #     # 添加噪声到信号
    #     received_signal = x + noise
    #
    #     return received_signal
    """加性高斯白噪声信道"""

    def forward(self, x, snr_db, **kwargs):
        # 方法1
        # 转换SNR从dB到线性尺度
        snr_linear = 10 ** (snr_db / 10)

        # 计算信号功率
        signal_power = torch.mean(torch.abs(x) ** 2)

        # 计算噪声功率
        noise_power = signal_power / snr_linear

        # 生成高斯噪声
        noise = torch.sqrt(noise_power) * torch.randn_like(x)

        # 添加噪声到信号
        received_signal = x + noise
        # # 方法2
        # # 信噪比从dB转换为线性尺度
        # snr = 10 ** (snr_db / 10)
        #
        # # 计算信号功率
        # xpower = torch.sum(torch.abs(x) ** 2) / x.numel()
        #
        # # 计算噪声功率
        # npower = xpower / snr
        #
        # # 生成高斯噪声
        # noise = torch.normal(0, torch.sqrt(npower), size=x.shape).to(device)
        #
        # # 添加噪声到信号
        # return x + noise

        # 方法3
        # received_signal = x + torch.normal(0, snr_db, size=x.shape).to(device)
        return received_signal


class RayleighChannel(Channel):
    """瑞利衰落信道"""

    def forward(self, x, snr_db, **kwargs):
        # 转换SNR从dB到线性尺度
        snr_linear = 10 ** (snr_db / 10)

        # 计算信号功率
        signal_power = torch.mean(torch.abs(x) ** 2)

        # 生成瑞利衰落系数 (复高斯随机变量的幅度)
        # 实部和虚部均为均值为0，方差为0.5的高斯分布
        h_real = torch.randn(x.shape, device=x.device) * math.sqrt(0.5)
        h_imag = torch.randn(x.shape, device=x.device) * math.sqrt(0.5)
        h = torch.sqrt(h_real ** 2 + h_imag ** 2)  # 瑞利分布

        # 应用衰落
        faded_signal = h * x

        # 计算噪声功率
        noise_power = signal_power / snr_linear

        # 生成高斯噪声
        noise = torch.sqrt(noise_power) * torch.randn_like(x)

        # 添加噪声到衰落后的信号
        received_signal = faded_signal + noise

        return received_signal


class RicianChannel(Channel):
    """莱斯衰落信道"""

    def forward(self, x, snr_db, K=1, **kwargs):
        # K因子: 直射路径功率与散射路径功率的比值
        K = kwargs.get('K', 1)  # 默认K=1

        # 转换SNR从dB到线性尺度
        snr_linear = 10 ** (snr_db / 10)

        # 计算信号功率
        signal_power = torch.mean(torch.abs(x) ** 2)

        # 生成莱斯衰落系数
        # 非中心参数
        v = math.sqrt(K / (K + 1))
        sigma = math.sqrt(1 / (2 * (K + 1)))

        # 生成高斯随机变量
        h_real = torch.randn(x.shape, device=x.device) * sigma + v
        h_imag = torch.randn(x.shape, device=x.device) * sigma
        h = torch.sqrt(h_real ** 2 + h_imag ** 2)  # 莱斯分布

        # 应用衰落
        faded_signal = h * x

        # 计算噪声功率
        noise_power = signal_power / snr_linear

        # 生成高斯噪声
        noise = torch.sqrt(noise_power) * torch.randn_like(x)

        # 添加噪声到衰落后的信号
        received_signal = faded_signal + noise

        return received_signal


class SemanticDecoder(nn.Module):
    def __init__(self, semantic_dim=SEMANTIC_DIM, vocab_size=30522):  # BERT vocab size
        super(SemanticDecoder, self).__init__()
        self.decoder_embedding = nn.Linear(semantic_dim, 768)
        self.bert = BertModel.from_pretrained('bert-base-german-cased')
        self.vocab_projection = nn.Linear(768, vocab_size)

    def forward(self, semantic_features, tgt_input_ids=None, tgt_attention_mask=None):
        # 扩展语义特征
        expanded_features = self.decoder_embedding(semantic_features)

        # 获取德语BERT的嵌入
        if tgt_input_ids is not None:
            outputs = self.bert(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask)
            last_hidden_state = outputs.last_hidden_state

            # 添加语义特征到最后隐藏状态
            # 将语义特征广播到所有token位置
            semantic_features_expanded = expanded_features.unsqueeze(1).expand(-1, last_hidden_state.size(1), -1)
            enhanced_hidden_state = last_hidden_state + semantic_features_expanded

            # 映射到词汇表大小
            logits = self.vocab_projection(enhanced_hidden_state)
            return logits
        else:
            # 推理模式 - 简化版本，实际应进行自回归解码
            # 这部分代码在实际翻译中需要更复杂的实现
            return expanded_features

        # 完整的语义通信系统


class SemanticCommunicationSystem(nn.Module):
    def __init__(self):
        super(SemanticCommunicationSystem, self).__init__()
        self.semantic_encoder = SemanticEncoder()
        self.channel_encoder = ChannelEncoder()
        self.channel_decoder = ChannelDecoder()
        self.semantic_decoder = SemanticDecoder()
        self.channels = {
            'awgn': AWGNChannel(),
            'rayleigh': RayleighChannel(),
            'rician': RicianChannel()
        }


    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids=None, tgt_attention_mask=None,
                snr_db=TRAINING_SNR, channel=None, channel_params={}):
        # 语义编码
        semantic_features = self.semantic_encoder(src_input_ids, src_attention_mask)

        # 信道编码
        channel_input = self.channel_encoder(semantic_features)

        # 通过规定信道（默认awgn）
        # channel_output = self.add_noise(channel_input, snr_db)
        if channel in self.channels:
            channel_model = self.channels[channel]
            channel_output = channel_model(channel_input, snr_db, **channel_params)
        else:
            # 默认使用AWGN
            channel_output = self.channels['awgn'](channel_input, snr_db)

            # 信道解码
        recovered_semantic = self.channel_decoder(channel_output)

        # 语义解码
        if tgt_input_ids is not None:
            output = self.semantic_decoder(recovered_semantic, tgt_input_ids, tgt_attention_mask)
            return output, recovered_semantic
        else:
            output = self.semantic_decoder(recovered_semantic)
            return output, recovered_semantic

        # --------------------- 3. 训练函数 ---------------------


def train_epoch(model, data_loader, optimizer, scheduler, tokenizer, criterion, epoch, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]")

    # 用于梯度累积的计数器
    accumulated_samples = 0
    optimizer.zero_grad()

    for i, batch in enumerate(progress_bar):
        # 将数据移到设备
        src_input_ids = batch['src_input_ids'].to(device)
        src_attention_mask = batch['src_attention_mask'].to(device)
        tgt_input_ids = batch['tgt_input_ids'].to(device)
        tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        # 使用混合精度训练
        # with autocast():
        with autocast(device_type='cuda'):
            # 前向传播
            logits, _ = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask,
                              channel=Channel_type)

            # 计算损失 (忽略padding token)
            # 移位目标序列用于训练（类似于teacher forcing）
            shifted_logits = logits[:, :-1].contiguous()  # 移除最后一个token的预测
            shifted_targets = tgt_input_ids[:, 1:].contiguous()  # 移除第一个token (通常是[CLS])
            shifted_mask = tgt_attention_mask[:, 1:].contiguous()

            # 计算交叉熵损失
            loss = criterion(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_targets.view(-1))
            loss = loss.view(shifted_targets.size())

            # 应用mask来忽略padding token
            masked_loss = loss * shifted_mask
            loss = masked_loss.sum() / shifted_mask.sum()

            # 梯度累积 - 缩放损失
            loss = loss / GRADIENT_ACCUMULATION_STEPS

            # 反向传播 - 使用scaler缩放梯度
        scaler.scale(loss).backward()

        # 更新总损失
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # 恢复原始损失值进行记录

        accumulated_samples += 1

        # 仅在梯度累积步数达到时更新参数
        if accumulated_samples % GRADIENT_ACCUMULATION_STEPS == 0 or i == len(progress_bar) - 1:
            # 梯度裁剪防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # 重置累积计数器
            accumulated_samples = 0

            # 更新进度条
        progress_bar.set_postfix({'loss': loss.item() * GRADIENT_ACCUMULATION_STEPS})

    return total_loss / len(data_loader)


def validate(model, data_loader, tokenizer, criterion, snr_db=TRAINING_SNR):
    model.eval()
    total_loss = 0
    bleu_scorer = BLEU()
    all_translations = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Validation [SNR={snr_db}dB]"):
            # 将数据移到设备
            src_input_ids = batch['src_input_ids'].to(device)
            src_attention_mask = batch['src_attention_mask'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            tgt_attention_mask = batch['tgt_attention_mask'].to(device)

            # 使用混合精度进行验证
            # with autocast():
            with autocast(device_type='cuda'):
                # 前向传播
                logits, _ = model(
                    src_input_ids,
                    src_attention_mask,
                    tgt_input_ids,
                    tgt_attention_mask,
                    snr_db=snr_db,
                    channel=Channel_type
                )

                # 计算损失
                shifted_logits = logits[:, :-1].contiguous()
                shifted_targets = tgt_input_ids[:, 1:].contiguous()
                shifted_mask = tgt_attention_mask[:, 1:].contiguous()

                loss = criterion(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_targets.view(-1))
                loss = loss.view(shifted_targets.size())
                masked_loss = loss * shifted_mask
                loss = masked_loss.sum() / shifted_mask.sum()

                total_loss += loss.item()

                # 生成翻译结果
            # 简化版本：使用argmax而非beam search来获取预测的token
            pred_tokens = torch.argmax(logits, dim=-1)

            # 将预测的token ID转换为文本
            for i in range(pred_tokens.size(0)):
                pred_text = tokenizer.decode(
                    pred_tokens[i].tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                ref_text = batch['tgt_text'][i]

                all_translations.append(pred_text)
                all_references.append([ref_text])  # 注意references需要是列表的列表

    # 计算BLEU分数(统一BLEU计算方法）
    bleu_score = bleu_scorer.corpus_score(all_translations, all_references).score

    return total_loss / len(data_loader), bleu_score


# --------------------- 4. 完整训练流程 ---------------------

def train_model():
    # 准备数据
    train_loader, valid_loader, test_loader, src_tokenizer, tgt_tokenizer = prepare_data()

    # 创建模型
    model = SemanticCommunicationSystem().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 学习率调度器
    # 考虑梯度累积的有效训练步数
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # 创建混合精度训练的scaler
    scaler = GradScaler()

    # 创建保存模型的目录
    os.makedirs('test_models', exist_ok=True)

    # 训练循环
    best_bleu = 0
    train_losses = []
    valid_losses = []
    bleu_scores = []

    print("Starting training...")
    print(f"使用设备: {device}")
    print(f"批次大小: {BATCH_SIZE}, 梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}, 有效批次大小: {effective_batch_size}")
    print(f"序列最大长度: {MAX_LENGTH}, 语义特征维度: {SEMANTIC_DIM}")

    for epoch in range(EPOCHS):
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, tgt_tokenizer, criterion, epoch, scaler)
        train_losses.append(train_loss)

        # 验证
        valid_loss, bleu_score = validate(model, valid_loader, tgt_tokenizer, criterion)
        valid_losses.append(valid_loss)
        bleu_scores.append(bleu_score)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, BLEU: {bleu_score:.2f}")
        save_metrics_to_csv(train_losses, valid_losses, bleu_scores, 'metrics/training_metrics.csv',channel_type = Channel_type)

        # 保存最佳模型
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save(model.state_dict(), f'test_models/best_model.pt')
            print(f"New best model saved with BLEU: {bleu_score:.2f}")

            # 每个epoch保存一次检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'bleu_score': bleu_score,
        }, f'test_models/checkpoint_epoch_{epoch + 1}.pt')

        # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(bleu_scores)
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score on Validation Set')

    plt.tight_layout()
    plt.savefig('training_curves3.png')


    return model, test_loader, tgt_tokenizer


# --------------------- 5. 在不同SNR下测试 ---------------------

def test_at_different_snrs(model, test_loader, tokenizer):
    # 加载最佳模型
    # model.load_state_dict(torch.load('test_models/best_model.pt', weights_only=True))
    # model.eval()

    # 加载最后模型
    # checkpoint = torch.load('test_models/checkpoint_epoch_21.pt', weights_only=True)  # 将X替换为您想加载的具体epoch
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()

    # 量化后的模型
    quantized_device = torch.device("cpu")

    # 正确方法：加载整个量化模型
    model = torch.load("quantized_full_model.pt")
    model = model.to(quantized_device)
    model.eval()


    # 存储不同SNR下的性能
    results = {}
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 在不同SNR下测试
    for snr_db in TEST_SNRS:
        valid_loss, bleu_score = validate(model, test_loader, tokenizer, criterion, snr_db=snr_db)
        results[snr_db] = {
            'loss': valid_loss,
            'bleu': bleu_score
        }
        print(f"SNR: {snr_db}dB - Loss: {valid_loss:.4f}, BLEU: {bleu_score:.2f}")
    df = pd.DataFrame(results)

    # 确保目录存在
    # os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    # 保存到CSV
    df.to_csv('metrics/test_metrics.csv', index=False)
    print(f"多SNR测试指标已保存到 {'metrics/test_metrics.csv'}")

    plt_snr_performance(results,channel_type=Channel_type)

    return results


# --------------------- 6. 性能分析 ---------------------

def analyze_model_performance(model):
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 分析每个组件的参数量
    se_params = sum(p.numel() for p in model.semantic_encoder.parameters())
    ce_params = sum(p.numel() for p in model.channel_encoder.parameters())
    cd_params = sum(p.numel() for p in model.channel_decoder.parameters())
    sd_params = sum(p.numel() for p in model.semantic_decoder.parameters())

    print("\nParameters by component:")
    print(f"Semantic Encoder: {se_params:,} ({se_params / total_params * 100:.1f}%)")
    print(f"Channel Encoder: {ce_params:,} ({ce_params / total_params * 100:.1f}%)")
    print(f"Channel Decoder: {cd_params:,} ({cd_params / total_params * 100:.1f}%)")
    print(f"Semantic Decoder: {sd_params:,} ({sd_params / total_params * 100:.1f}%)")


# --------------------- 7. 主函数 ---------------------

def main():
    # 训练模型
    model, test_loader, tokenizer = train_model()

    # 分析模型性能
    analyze_model_performance(model)

    # 验证test_at_different_snr函数
    # model = SemanticCommunicationSystem().to(device)
    # train_loader, valid_loader, test_loader, src_tokenizer, tokenizer = prepare_data()
    # 在不同SNR下测试性能
    results = test_at_different_snrs(model, test_loader, tokenizer)
    # results = test_at_different_snrs(model, valid_loader, tokenizer)
    print("Training and evaluation completed!")
    return model, results


if __name__ == "__main__":
    main()

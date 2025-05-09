import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datasets import load_dataset
from tqdm import tqdm
import nltk
from nltk.util import ngrams
import os

# 创建输出目录
os.makedirs('dataset_analysis', exist_ok=True)

# 下载NLTK资源
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def analyze_iwslt17_deen():
    """全面分析IWSLT17德英数据集"""
    print("正在加载IWSLT17德英数据集...")

    # 加载数据集
    dataset = load_dataset("iwslt2017", "iwslt2017-de-en")

    # 1. 基本统计分析
    print("\n==== 基本数据集信息 ====")
    for split in dataset.keys():
        print(f"{split}集: {len(dataset[split])}个样本")

        # 2. 数据分布分析
    analyze_length_distribution(dataset)

    # 3. 词汇分析
    analyze_vocabulary(dataset)

    # 4. 句子复杂度分析
    analyze_sentence_complexity(dataset)

    # 5. 平行语料特性分析
    analyze_parallel_characteristics(dataset)

    # 6. 导出样本
    export_samples(dataset)

    print("\n分析完成! 结果已保存到 'dataset_analysis' 目录")


def analyze_length_distribution(dataset):
    """分析句子长度分布"""
    print("\n==== 句子长度分布分析 ====")

    # 收集所有数据的句子长度
    splits = ['train', 'validation', 'test']
    de_lengths = {split: [] for split in splits}
    en_lengths = {split: [] for split in splits}

    for split in splits:
        for item in tqdm(dataset[split], desc=f"处理{split}集"):
            de_lengths[split].append(len(item['translation']['de'].split()))
            en_lengths[split].append(len(item['translation']['en'].split()))

            # 计算统计指标
    for split in splits:
        print(f"\n{split}集统计:")
        print(f"  德语句子长度: 最小={min(de_lengths[split])}, 最大={max(de_lengths[split])}, "
              f"平均={np.mean(de_lengths[split]):.2f}, 中位数={np.median(de_lengths[split])}")
        print(f"  英语句子长度: 最小={min(en_lengths[split])}, 最大={max(en_lengths[split])}, "
              f"平均={np.mean(en_lengths[split]):.2f}, 中位数={np.median(en_lengths[split])}")

        # 生成长度分布图
    plt.figure(figsize=(12, 8))
    sns.histplot(de_lengths['train'], kde=True, color='blue', alpha=0.5, label='德语')
    sns.histplot(en_lengths['train'], kde=True, color='red', alpha=0.5, label='英语')
    plt.title('训练集句子长度分布')
    plt.xlabel('单词数')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('dataset_analysis/sentence_length_distribution.png', dpi=300, bbox_inches='tight')

    # 计算长句比例
    long_de = sum(1 for l in de_lengths['train'] if l > 40) / len(de_lengths['train']) * 100
    long_en = sum(1 for l in en_lengths['train'] if l > 40) / len(en_lengths['train']) * 100
    print(f"\n超过40个单词的句子比例: 德语={long_de:.2f}%, 英语={long_en:.2f}%")

    # 长度比例分析
    length_ratios = [en_len / de_len if de_len > 0 else 0
                     for de_len, en_len in zip(de_lengths['train'], en_lengths['train'])]
    avg_ratio = np.mean([r for r in length_ratios if r > 0])
    print(f"英语/德语平均长度比: {avg_ratio:.3f}")

    plt.figure(figsize=(10, 6))
    sns.histplot(length_ratios, kde=True)
    plt.title('英语/德语句子长度比例分布')
    plt.xlabel('长度比例 (英语/德语)')
    plt.ylabel('频率')
    plt.axvline(avg_ratio, color='red', linestyle='--', label=f'平均比例: {avg_ratio:.3f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('dataset_analysis/length_ratio_distribution.png', dpi=300, bbox_inches='tight')


def analyze_vocabulary(dataset):
    """分析词汇特性"""
    print("\n==== 词汇分析 ====")

    # 构建词汇表
    de_vocab = Counter()
    en_vocab = Counter()

    for item in tqdm(dataset['train'], desc="构建词汇表"):
        de_words = item['translation']['de'].lower().split()
        en_words = item['translation']['en'].lower().split()
        de_vocab.update(de_words)
        en_vocab.update(en_words)

    print(f"德语词汇量: {len(de_vocab)} 独立单词")
    print(f"英语词汇量: {len(en_vocab)} 独立单词")

    # 分析词频分布
    de_freq = sorted(de_vocab.values(), reverse=True)
    en_freq = sorted(en_vocab.values(), reverse=True)

    # 计算覆盖率
    de_total = sum(de_freq)
    en_total = sum(en_freq)

    de_coverage = [sum(de_freq[:i]) / de_total for i in [1000, 5000, 10000, 20000]]
    en_coverage = [sum(en_freq[:i]) / en_total for i in [1000, 5000, 10000, 20000]]

    print("\n词汇覆盖率:")
    for i, n in enumerate([1000, 5000, 10000, 20000]):
        print(f"  前{n}个常见词: 德语={de_coverage[i] * 100:.2f}%, 英语={en_coverage[i] * 100:.2f}%")

        # 词频分布图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.loglog(range(1, len(de_freq) + 1), de_freq, 'b-')
    plt.title('德语词频分布 (对数尺度)')
    plt.xlabel('词频排名')
    plt.ylabel('频率')
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.loglog(range(1, len(en_freq) + 1), en_freq, 'r-')
    plt.title('英语词频分布 (对数尺度)')
    plt.xlabel('词频排名')
    plt.ylabel('频率')
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig('dataset_analysis/vocabulary_distribution.png', dpi=300, bbox_inches='tight')

    # 常见词分析
    print("\n最常见的50个词:")
    print("  德语:", ", ".join([word for word, _ in de_vocab.most_common(50)]))
    print("  英语:", ", ".join([word for word, _ in en_vocab.most_common(50)]))

    # 稀有词分析
    de_rare = [word for word, count in de_vocab.items() if count == 1]
    en_rare = [word for word, count in en_vocab.items() if count == 1]

    print(f"\n只出现一次的词 (hapax legomena):")
    print(f"  德语: {len(de_rare)}个 ({len(de_rare) / len(de_vocab) * 100:.2f}%)")
    print(f"  英语: {len(en_rare)}个 ({len(en_rare) / len(en_vocab) * 100:.2f}%)")


def analyze_sentence_complexity(dataset):
    """分析句子复杂度"""
    print("\n==== 句子复杂度分析 ====")

    # 抽样分析
    sample_size = min(10000, len(dataset['train']))
    sample_indices = np.random.choice(len(dataset['train']), sample_size, replace=False)

    de_ngrams = {2: Counter(), 3: Counter()}
    en_ngrams = {2: Counter(), 3: Counter()}

    for idx in tqdm(sample_indices, desc="分析句子复杂度"):
        item = dataset['train'][idx]

        # 提取n-grams
        de_tokens = nltk.word_tokenize(item['translation']['de'].lower())
        en_tokens = nltk.word_tokenize(item['translation']['en'].lower())

        for n in [2, 3]:
            de_ngrams[n].update(ngrams(de_tokens, n))
            en_ngrams[n].update(ngrams(en_tokens, n))

            # 打印最常见的n-grams
    print("\n最常见的Bigrams (2-grams):")
    print("  德语:", ", ".join([" ".join(gram) for gram, _ in de_ngrams[2].most_common(10)]))
    print("  英语:", ", ".join([" ".join(gram) for gram, _ in en_ngrams[2].most_common(10)]))

    print("\n最常见的Trigrams (3-grams):")
    print("  德语:", ", ".join([" ".join(gram) for gram, _ in de_ngrams[3].most_common(10)]))
    print("  英语:", ", ".join([" ".join(gram) for gram, _ in en_ngrams[3].most_common(10)]))

    # 句子结构分析
    de_punct = Counter()
    en_punct = Counter()
    question_de = 0
    question_en = 0

    for idx in tqdm(sample_indices, desc="分析句子结构"):
        item = dataset['train'][idx]

        de_text = item['translation']['de']
        en_text = item['translation']['en']

        de_punct.update(re.findall(r'[,.;:!?]', de_text))
        en_punct.update(re.findall(r'[,.;:!?]', en_text))

        if '?' in de_text:
            question_de += 1
        if '?' in en_text:
            question_en += 1

    print("\n标点符号分布:")
    print(f"  德语: {dict(de_punct)}")
    print(f"  英语: {dict(en_punct)}")

    print(f"\n疑问句比例:")
    print(f"  德语: {question_de / sample_size * 100:.2f}%")
    print(f"  英语: {question_en / sample_size * 100:.2f}%")


def analyze_parallel_characteristics(dataset):
    """分析平行语料的特性"""
    print("\n==== 平行语料特性分析 ====")

    # 抽样分析
    sample_size = min(5000, len(dataset['train']))
    sample_indices = np.random.choice(len(dataset['train']), sample_size, replace=False)

    length_diffs = []
    digit_matches = []
    name_matches = []

    for idx in tqdm(sample_indices, desc="分析平行特性"):
        item = dataset['train'][idx]

        de_text = item['translation']['de']
        en_text = item['translation']['en']

        # 长度差异
        de_len = len(de_text.split())
        en_len = len(en_text.split())
        length_diff = abs(de_len - en_len) / max(de_len, en_len)
        length_diffs.append(length_diff)

        # 数字匹配度
        de_digits = re.findall(r'\d+', de_text)
        en_digits = re.findall(r'\d+', en_text)

        if de_digits or en_digits:
            common = set(de_digits) & set(en_digits)
            total = set(de_digits) | set(en_digits)
            digit_matches.append(len(common) / len(total) if total else 1.0)

            # 专有名词匹配度 (简化版)
        de_caps = set(re.findall(r'\b[A-Z][a-zA-Z]*\b', de_text))
        en_caps = set(re.findall(r'\b[A-Z][a-zA-Z]*\b', en_text))

        if de_caps or en_caps:
            common = set(w.lower() for w in de_caps) & set(w.lower() for w in en_caps)
            total = set(w.lower() for w in de_caps) | set(w.lower() for w in en_caps)
            name_matches.append(len(common) / len(total) if total else 1.0)

            # 输出结果
    print(f"平均相对长度差异: {np.mean(length_diffs):.4f}")
    print(f"数字匹配率: {np.mean(digit_matches):.4f}")
    print(f"专有名词匹配率: {np.mean(name_matches):.4f}")

    # 绘制长度差异分布
    plt.figure(figsize=(10, 6))
    sns.histplot(length_diffs, kde=True)
    plt.title('德英句子对相对长度差异分布')
    plt.xlabel('相对长度差异 |de_len - en_len| / max(de_len, en_len)')
    plt.ylabel('频率')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('dataset_analysis/length_difference_distribution.png', dpi=300, bbox_inches='tight')

    # 可能的对齐问题分析
    potential_alignment_issues = sum(1 for diff in length_diffs if diff > 0.5)
    print(f"潜在对齐问题的句对比例: {potential_alignment_issues / len(length_diffs) * 100:.2f}%")


def export_samples(dataset):
    """导出样本数据"""
    # 随机抽取100个样本
    train_samples = dataset['train'].select(range(100))

    # 创建DataFrame
    samples_df = pd.DataFrame({
        'German': [item['translation']['de'] for item in train_samples],
        'English': [item['translation']['en'] for item in train_samples]
    })

    # 导出到CSV
    samples_df.to_csv('dataset_analysis/sample_data.csv', index=False, encoding='utf-8')

    # 导出长句样本
    long_samples = []
    for item in dataset['train']:
        de_len = len(item['translation']['de'].split())
        en_len = len(item['translation']['en'].split())
        if de_len > 40 or en_len > 40:
            long_samples.append(item)
            if len(long_samples) >= 20:
                break

    long_df = pd.DataFrame({
        'German': [item['translation']['de'] for item in long_samples],
        'English': [item['translation']['en'] for item in long_samples],
        'German_Length': [len(item['translation']['de'].split()) for item in long_samples],
        'English_Length': [len(item['translation']['en'].split()) for item in long_samples]
    })

    long_df.to_csv('dataset_analysis/long_sentence_samples.csv', index=False, encoding='utf-8')


if __name__ == "__main__":
    analyze_iwslt17_deen()
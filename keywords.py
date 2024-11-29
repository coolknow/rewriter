import json
import re
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
import math

# 下载 NLTK 停用词列表
# nltk.download('stopwords')

# 获取英语停用词列表
stop_words = set(stopwords.words('english'))

# 读取数据
def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def preprocess_text(text):
    # 去除特定标记（如 LaTeX 的 textbf, textit 等）
    text = re.sub(r'textbf|textit|\\[a-z]+', '', text)
    # 去除非字母、数字、连字符和空格，保留字母开头的单词
    text = re.sub(r'[^a-zA-Z0-9\- ]', '', text)
    # 分词
    words = text.split()
    # 仅保留以英文字母开头的单词，并去除停用词
    filtered_words = [word.lower() for word in words if re.match(r'^[a-zA-Z]', word) and word.lower() not in stop_words]
    return filtered_words

# 提取所有的 filtered_words 并统计词频
def extract_filtered_word_frequencies(data):
    word_counter = Counter()  # 统计每个单词的频次
    for entry in data:
        abstract = entry.get("abstract", "")
        words = preprocess_text(abstract)
        word_counter.update(words)  # 更新词频
    return word_counter

# 统计词频的频次（词频的词频）
def get_frequency_distribution(word_counter):
    # 获取每个词的频次
    frequency_counter = Counter(word_counter.values())  # 统计每个频次的出现次数
    return frequency_counter

# 找出最接近目标频率的代表词
def find_closest_words(word_counter, target_freqs):
    closest_words = {}
    sorted_word_freqs = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)  # 按频率排序
    
    for target in target_freqs:
        closest_word = min(sorted_word_freqs, key=lambda x: abs(x[1] - target))  # 找到最接近目标频率的词
        closest_words[target] = closest_word
    return closest_words

# 绘制词频分布图
def plot_frequency_distribution(frequency_counter, closest_words):
    # 过滤出频率在 0 到 500 之间的数据
    filtered_data = {freq: count for freq, count in frequency_counter.items() if 0 <= freq <= 100}
    frequencies, counts = zip(*sorted(filtered_data.items()))
    
    # 绘制频率分布图
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, counts, marker='o', linestyle='-', color='b', label='Word Frequency Distribution (0-100)')
    
    # 在图上标记代表单词
    for target, (word, freq) in closest_words.items():
        if freq <= 100:  # 仅标记频率 <= 500 的单词
            # 获取当前点的横纵坐标
            x, y = freq, filtered_data[freq]
            # 偏移量计算，距离可以调节（如：1000, 10 是距离）
            offset_distance = 50  # 调整偏移距离，适配小范围的频率
            dx = offset_distance * math.cos(math.radians(90))  # 水平方向偏移
            dy = 10 * offset_distance * math.sin(math.radians(90))  # 垂直方向偏移
            plt.annotate(f'{word} ({freq})', 
                         xy=(x, y),  # 箭头指向的点
                         xytext=(x + dx, y + dy),  # 标注的位置
                         arrowprops=dict(facecolor='red', arrowstyle='->'),
                         fontsize=9, color='red')

    plt.xlabel('Word Frequency')  # 词频
    plt.ylabel('Number of Words with that Frequency')  # 具有该词频的单词数
    plt.title('Filtered Word Frequency Distribution (0-100)')
    plt.grid(True)
    plt.legend()
    plt.show()


# 主程序
if __name__ == "__main__":
    input_file = 'neurips_2023/abstracts.jsonl'  # 你的数据文件
    output_file = 'neurips_2023/vocabulary_with_freq_exclude_stopwords.jsonl'  # 输出词汇表的文件
    data = read_jsonl(input_file)
    
    # 提取过滤后的词频
    filtered_word_frequencies = extract_filtered_word_frequencies(data)
    
    # 统计词频的词频
    frequency_distribution = get_frequency_distribution(filtered_word_frequencies)
    
    # 找到最接近 5000, 10000, 15000, ... 频次的代表单词
    target_freqs = [20, 40, 60, 80]
    closest_words = find_closest_words(filtered_word_frequencies, target_freqs)
    
    # 保存词汇表
    with open(output_file, 'w') as f:
        for word, count in sorted(filtered_word_frequencies.items()):
            json.dump({"word": word, "count": count}, f)
            f.write('\n')
    
    # 绘制词频分布图
    plot_frequency_distribution(frequency_distribution, closest_words)

    print(f"词汇表已保存到 {output_file}")

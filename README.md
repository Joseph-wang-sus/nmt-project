# NMT Project - 神经机器翻译项目

[English](#english-version) | [中文](#中文版本)

---

## 中文版本

### 项目简介

这是一个完整的中英神经机器翻译（Neural Machine Translation, NMT）项目，实现了多种主流的序列到序列模型架构，包括：

- **RNN-based 模型**：支持 LSTM 和 GRU，配备多种注意力机制（Dot、General、Additive）
- **Transformer 模型**：包含位置编码和归一化的消融实验
- **T5 微调**：基于预训练的 T5/mT5 模型进行迁移学习

### 主要功能

- ✅ 完整的数据预处理管道（清洗、分词、词表构建）
- ✅ 多种模型架构实现和训练
- ✅ 灵活的解码策略（贪婪搜索、束搜索）
- ✅ 自动化实验管理和批量推理
- ✅ BLEU 评估指标
- ✅ 支持超参数网格搜索

### 项目结构

```
nmt-project/
├── models/                      # 模型定义
│   ├── rnn_nmt.py              # RNN 编码器-解码器
│   ├── transformer_nmt.py      # Transformer 模型
│   └── attention.py            # 注意力机制实现
├── preprocess.py               # 数据预处理脚本
├── data_utils.py               # 数据处理工具
├── train_rnn.py                # RNN 模型训练
├── train_transformer.py        # Transformer 模型训练
├── fine_tune_t5.py             # T5 模型微调
├── inference_rnn.py            # RNN 推理
├── inference_transformer.py    # Transformer 推理
├── run_all_inference.py        # 批量推理评估
├── run_pipeline.sh             # 完整实验流程
├── decoding_utils.py           # 解码工具（贪婪、束搜索）
├── transformer_decoding.py     # Transformer 专用解码
└── requirements.txt            # Python 依赖
```

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA（推荐用于 GPU 加速）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/Joseph-wang-sus/nmt-project.git
cd nmt-project
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

依赖包括：
- `torch>=2.0.0` - PyTorch 深度学习框架
- `transformers>=4.0.0` - Hugging Face Transformers
- `jieba>=0.42.1` - 中文分词
- `sacrebleu>=2.0.0` - BLEU 评估
- `hanlp` - 中文 NLP 工具（通过 data_utils.py 使用）
- `tokenizers` - BPE 分词器

### 使用方法

#### 1. 数据准备

准备 JSONL 格式的训练和验证数据，每行包含 `zh` 和 `en` 字段：

```json
{"zh": "你好世界", "en": "Hello world", "index": 0}
{"zh": "机器翻译", "en": "Machine translation", "index": 1}
```

将数据文件放置在 `data/` 目录下：
- `data/train_100k.jsonl` - 训练集
- `data/valid.jsonl` - 验证集

#### 2. 数据预处理

运行预处理脚本进行数据清洗、分词和词表构建：

```bash
python preprocess.py \
  --train_file data/train_100k.jsonl \
  --valid_file data/valid.jsonl \
  --output_dir data_processed \
  --vocab_size_en 16000 \
  --min_freq_zh 2 \
  --max_len 160
```

参数说明：
- `--vocab_size_en`: 英文 BPE 词表大小
- `--min_freq_zh`: 中文词最小频率
- `--max_len`: 最大序列长度

输出文件：
- `data_processed/processed_data.pt` - 处理后的数据和词表
- `data_processed/tokenizer_en.json` - 英文 BPE 分词器

#### 3. 模型训练

##### 训练 Transformer 模型

```bash
python train_transformer.py \
  --data_path data_processed/processed_data.pt \
  --save_path checkpoints/transformer_exp1 \
  --d_model 256 \
  --n_heads 8 \
  --n_layers 3 \
  --batch_size 128 \
  --lr 5e-4 \
  --epochs 10 \
  --norm_type layernorm \
  --pos_type absolute
```

关键参数：
- `--d_model`: 模型维度（128, 256, 512）
- `--n_heads`: 注意力头数量
- `--n_layers`: 编码器/解码器层数
- `--norm_type`: 归一化类型（layernorm 或 rmsnorm）
- `--pos_type`: 位置编码类型（absolute 或 relative）

##### 训练 RNN 模型

```bash
python train_rnn.py \
  --data_path data_processed/processed_data.pt \
  --save_dir checkpoints/rnn_exp1 \
  --rnn_type gru \
  --attention_type dot \
  --hidden_size 256 \
  --embed_size 256 \
  --batch_size 96 \
  --lr 0.001 \
  --epochs 10 \
  --teacher_forcing 0.5
```

关键参数：
- `--rnn_type`: RNN 类型（gru 或 lstm）
- `--attention_type`: 注意力类型（dot、general 或 additive）
- `--hidden_size`: 隐藏层大小
- `--teacher_forcing`: Teacher forcing 比例

##### 微调 T5 模型

```bash
python fine_tune_t5.py \
  --train_file data/train_100k.jsonl \
  --valid_file data/valid.jsonl \
  --model_name google/mt5-small \
  --output_dir t5_result \
  --batch_size 16 \
  --epochs 3 \
  --lr 2e-4
```

#### 4. 模型推理

##### Transformer 推理

```bash
python inference_transformer.py \
  --checkpoint checkpoints/transformer_exp1/best_model.pt \
  --data_path data_processed/processed_data.pt \
  --decode_method beam \
  --beam_size 5 \
  --max_len 100
```

##### RNN 推理

```bash
python inference_rnn.py \
  --checkpoint checkpoints/rnn_exp1/best_model.pt \
  --data_path data_processed/processed_data.pt \
  --decode_method beam \
  --beam_size 5
```

##### T5 推理

```bash
python run_t5_inference.py \
  --model_dir t5_result \
  --test_file data/valid.jsonl \
  --output_file t5_predictions.txt
```

#### 5. 批量实验和评估

##### 运行完整实验管道

`run_pipeline.sh` 脚本自动化执行预处理和多组超参数实验：

```bash
bash run_pipeline.sh
```

该脚本会：
1. 运行数据预处理
2. 启动 Transformer 超参数网格搜索（不同的 d_model、层数、学习率）
3. 启动 RNN 超参数网格搜索（不同的 RNN 类型、注意力机制、隐藏层大小）

##### 批量推理评估

评估所有训练好的模型：

```bash
python run_all_inference.py \
  --checkpoints_root checkpoints \
  --data_path data_processed/processed_data.pt \
  --output_csv results.csv
```

### 模型架构说明

#### RNN-based 模型

- **编码器**：双向 LSTM/GRU
- **解码器**：单向 LSTM/GRU 配合注意力机制
- **注意力机制**：
  - Dot Attention: `score(h_t, h_s) = h_t^T h_s`
  - General Attention: `score(h_t, h_s) = h_t^T W h_s`
  - Additive Attention: `score(h_t, h_s) = v^T tanh(W1 h_t + W2 h_s)`

#### Transformer 模型

- **标准架构**：Multi-head Self-Attention + Feed-Forward Networks
- **位置编码**：
  - Absolute: 正弦/余弦位置编码
  - Relative: 相对位置编码
- **归一化**：
  - LayerNorm: 标准层归一化
  - RMSNorm: Root Mean Square Layer Normalization

#### T5 模型

- 基于预训练的 `google/t5-small` 或 `google/mt5-small`
- 使用 "translate Chinese to English:" 作为任务前缀
- 支持迁移学习和领域适应

### 评估指标

使用 **SacreBLEU** 进行标准化的 BLEU 分数计算，确保结果可重现和可比较。

### 实验配置示例

#### Transformer 网格搜索

```bash
d_model: [128, 256]
n_layers: [2, 3]
learning_rate: [5e-4, 1e-3]
n_heads: 根据 d_model 自动计算
```

#### RNN 网格搜索

```bash
rnn_type: [gru, lstm]
attention_type: [dot, general]
hidden_size: [256, 384]
teacher_forcing: [0.5, 0.7]
```

### 性能优化建议

1. **GPU 加速**：所有训练脚本自动检测并使用 CUDA
2. **批次大小**：根据 GPU 内存调整（Transformer: 64-128, RNN: 64-96）
3. **序列长度**：使用动态填充减少计算量
4. **教师强制**：RNN 模型中逐步降低可提升泛化能力
5. **学习率调度**：建议使用 warmup + 余弦退火

### 常见问题

**Q: 如何处理中文分词？**  
A: 项目使用 HanLP 进行中文分词，首次运行会自动下载模型。

**Q: 内存不足怎么办？**  
A: 减小 `batch_size` 或 `max_len` 参数。

**Q: 如何选择模型？**  
A: Transformer 通常性能更好但训练较慢；RNN 训练快但可能需要更多调参；T5 适合小数据集。

**Q: BLEU 分数偏低？**  
A: 检查数据质量、增加训练轮数、调整学习率或使用更大的模型。

### 许可证

本项目遵循 MIT 许可证。

### 贡献

欢迎提交 Issue 和 Pull Request！

---

## English Version

### Introduction

A comprehensive Neural Machine Translation (NMT) project for Chinese-to-English translation, implementing multiple state-of-the-art sequence-to-sequence architectures:

- **RNN-based Models**: LSTM and GRU with multiple attention mechanisms (Dot, General, Additive)
- **Transformer Models**: With ablation studies on positional encoding and normalization
- **T5 Fine-tuning**: Transfer learning with pretrained T5/mT5 models

### Key Features

- ✅ Complete data preprocessing pipeline (cleaning, tokenization, vocabulary building)
- ✅ Multiple model architectures with training scripts
- ✅ Flexible decoding strategies (greedy search, beam search)
- ✅ Automated experiment management and batch inference
- ✅ BLEU evaluation metrics
- ✅ Hyperparameter grid search support

### Project Structure

```
nmt-project/
├── models/                      # Model definitions
│   ├── rnn_nmt.py              # RNN encoder-decoder
│   ├── transformer_nmt.py      # Transformer model
│   └── attention.py            # Attention mechanisms
├── preprocess.py               # Data preprocessing
├── data_utils.py               # Data utilities
├── train_rnn.py                # RNN training
├── train_transformer.py        # Transformer training
├── fine_tune_t5.py             # T5 fine-tuning
├── inference_rnn.py            # RNN inference
├── inference_transformer.py    # Transformer inference
├── run_all_inference.py        # Batch inference evaluation
├── run_pipeline.sh             # Complete experiment pipeline
├── decoding_utils.py           # Decoding utilities
├── transformer_decoding.py     # Transformer-specific decoding
└── requirements.txt            # Python dependencies
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (recommended for GPU acceleration)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Joseph-wang-sus/nmt-project.git
cd nmt-project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Dependencies include:
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.0.0` - Hugging Face Transformers
- `jieba>=0.42.1` - Chinese word segmentation
- `sacrebleu>=2.0.0` - BLEU evaluation
- `hanlp` - Chinese NLP toolkit
- `tokenizers` - BPE tokenizer

### Usage

#### 1. Data Preparation

Prepare training and validation data in JSONL format with `zh` and `en` fields:

```json
{"zh": "你好世界", "en": "Hello world", "index": 0}
{"zh": "机器翻译", "en": "Machine translation", "index": 1}
```

Place data files in the `data/` directory:
- `data/train_100k.jsonl` - Training set
- `data/valid.jsonl` - Validation set

#### 2. Data Preprocessing

Run preprocessing for data cleaning, tokenization, and vocabulary building:

```bash
python preprocess.py \
  --train_file data/train_100k.jsonl \
  --valid_file data/valid.jsonl \
  --output_dir data_processed \
  --vocab_size_en 16000 \
  --min_freq_zh 2 \
  --max_len 160
```

Parameters:
- `--vocab_size_en`: English BPE vocabulary size
- `--min_freq_zh`: Minimum frequency for Chinese tokens
- `--max_len`: Maximum sequence length

Output files:
- `data_processed/processed_data.pt` - Processed data and vocabularies
- `data_processed/tokenizer_en.json` - English BPE tokenizer

#### 3. Model Training

##### Train Transformer Model

```bash
python train_transformer.py \
  --data_path data_processed/processed_data.pt \
  --save_path checkpoints/transformer_exp1 \
  --d_model 256 \
  --n_heads 8 \
  --n_layers 3 \
  --batch_size 128 \
  --lr 5e-4 \
  --epochs 10 \
  --norm_type layernorm \
  --pos_type absolute
```

Key parameters:
- `--d_model`: Model dimension (128, 256, 512)
- `--n_heads`: Number of attention heads
- `--n_layers`: Number of encoder/decoder layers
- `--norm_type`: Normalization type (layernorm or rmsnorm)
- `--pos_type`: Positional encoding type (absolute or relative)

##### Train RNN Model

```bash
python train_rnn.py \
  --data_path data_processed/processed_data.pt \
  --save_dir checkpoints/rnn_exp1 \
  --rnn_type gru \
  --attention_type dot \
  --hidden_size 256 \
  --embed_size 256 \
  --batch_size 96 \
  --lr 0.001 \
  --epochs 10 \
  --teacher_forcing 0.5
```

Key parameters:
- `--rnn_type`: RNN type (gru or lstm)
- `--attention_type`: Attention type (dot, general, or additive)
- `--hidden_size`: Hidden layer size
- `--teacher_forcing`: Teacher forcing ratio

##### Fine-tune T5 Model

```bash
python fine_tune_t5.py \
  --train_file data/train_100k.jsonl \
  --valid_file data/valid.jsonl \
  --model_name google/mt5-small \
  --output_dir t5_result \
  --batch_size 16 \
  --epochs 3 \
  --lr 2e-4
```

#### 4. Model Inference

##### Transformer Inference

```bash
python inference_transformer.py \
  --checkpoint checkpoints/transformer_exp1/best_model.pt \
  --data_path data_processed/processed_data.pt \
  --decode_method beam \
  --beam_size 5 \
  --max_len 100
```

##### RNN Inference

```bash
python inference_rnn.py \
  --checkpoint checkpoints/rnn_exp1/best_model.pt \
  --data_path data_processed/processed_data.pt \
  --decode_method beam \
  --beam_size 5
```

##### T5 Inference

```bash
python run_t5_inference.py \
  --model_dir t5_result \
  --test_file data/valid.jsonl \
  --output_file t5_predictions.txt
```

#### 5. Batch Experiments and Evaluation

##### Run Complete Experiment Pipeline

The `run_pipeline.sh` script automates preprocessing and hyperparameter grid search:

```bash
bash run_pipeline.sh
```

This script will:
1. Run data preprocessing
2. Launch Transformer hyperparameter grid search (varying d_model, layers, learning rate)
3. Launch RNN hyperparameter grid search (varying RNN type, attention, hidden size)

##### Batch Inference Evaluation

Evaluate all trained models:

```bash
python run_all_inference.py \
  --checkpoints_root checkpoints \
  --data_path data_processed/processed_data.pt \
  --output_csv results.csv
```

### Model Architectures

#### RNN-based Models

- **Encoder**: Bidirectional LSTM/GRU
- **Decoder**: Unidirectional LSTM/GRU with attention mechanism
- **Attention Mechanisms**:
  - Dot Attention: `score(h_t, h_s) = h_t^T h_s`
  - General Attention: `score(h_t, h_s) = h_t^T W h_s`
  - Additive Attention: `score(h_t, h_s) = v^T tanh(W1 h_t + W2 h_s)`

#### Transformer Models

- **Standard Architecture**: Multi-head Self-Attention + Feed-Forward Networks
- **Positional Encoding**:
  - Absolute: Sinusoidal positional encoding
  - Relative: Relative positional encoding
- **Normalization**:
  - LayerNorm: Standard layer normalization
  - RMSNorm: Root Mean Square Layer Normalization

#### T5 Models

- Based on pretrained `google/t5-small` or `google/mt5-small`
- Uses "translate Chinese to English:" as task prefix
- Supports transfer learning and domain adaptation

### Evaluation Metrics

Uses **SacreBLEU** for standardized BLEU score calculation, ensuring reproducible and comparable results.

### Example Experiment Configurations

#### Transformer Grid Search

```bash
d_model: [128, 256]
n_layers: [2, 3]
learning_rate: [5e-4, 1e-3]
n_heads: Automatically calculated based on d_model
```

#### RNN Grid Search

```bash
rnn_type: [gru, lstm]
attention_type: [dot, general]
hidden_size: [256, 384]
teacher_forcing: [0.5, 0.7]
```

### Performance Optimization Tips

1. **GPU Acceleration**: All training scripts automatically detect and use CUDA
2. **Batch Size**: Adjust based on GPU memory (Transformer: 64-128, RNN: 64-96)
3. **Sequence Length**: Use dynamic padding to reduce computation
4. **Teacher Forcing**: Gradually decrease in RNN models to improve generalization
5. **Learning Rate Scheduling**: Recommend warmup + cosine annealing

### FAQ

**Q: How is Chinese tokenization handled?**  
A: The project uses HanLP for Chinese tokenization. Models are automatically downloaded on first run.

**Q: What if I run out of memory?**  
A: Reduce the `batch_size` or `max_len` parameters.

**Q: Which model should I choose?**  
A: Transformers usually perform better but train slower; RNNs train faster but may need more tuning; T5 is good for small datasets.

**Q: Why is my BLEU score low?**  
A: Check data quality, increase training epochs, adjust learning rate, or use larger models.

### License

This project is licensed under the MIT License.

### Contributing

Issues and Pull Requests are welcome!

---

### Citation

If you use this project in your research, please cite:

```bibtex
@misc{nmt-project,
  author = {Joseph Wang},
  title = {NMT Project: Neural Machine Translation for Chinese-English},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Joseph-wang-sus/nmt-project}
}
```

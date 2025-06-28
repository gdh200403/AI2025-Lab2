import collections
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from collections import Counter
import re
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from matplotlib.ticker import MaxNLocator

class Config:
    """用于存储超参数和路径的全局配置类"""

    SEED = 1234  
    MAX_LENGTH = 256  
    MIN_FREQ = 5  
    TEST_SIZE = 0.25  
    VOCAB_SPECIALS = ["<unk>", "<pad>"]  
    OUTPUT_DIM = 2  
    N_EPOCHS = 10  
    DATA_DIR = "./data/"  
    MODEL_DIR = "./models/"  
    FIGURE_DIR = "./figures/"      
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")  

    # 可调超参数（用于实验）
    # TODO: 对以下超参数进行不同值的实验：
    # - BATCH_SIZE: 尝试 [128, 256, 512] 以评估对训练速度和稳定性的影响，如果无显卡，可以尝试再调小batch_size
    # - EMBEDDING_DIM: 可尝试 [100, 200, 300] 以评估对模型容量和性能的影响
    # - HIDDEN_DIM: 可尝试 [100, 200, 300] 以研究对RNN表达能力的影响
    # - N_LAYERS: 可尝试 [1, 2, 3] 以探索深度和过拟合之间的权衡
    # - BIDIRECTIONAL: 可尝试 [True, False] 以比较双向与单向RNN的性能
    # - DROPOUT_RATE: 可尝试 [0.3, 0.5, 0.7] 以分析正则化效果
    # - LR: 可尝试 [1e-3, 5e-4, 1e-4] 以优化收敛速度和稳定性
    # 报告测试过的超参在测试集上的准确率和F1分数，并讨论权衡
    
    BATCH_SIZE = 128  # 较小的batch_size适合CPU训练
    EMBEDDING_DIM = 200  # 中等大小的嵌入维度
    HIDDEN_DIM = 200  # 中等大小的隐藏维度
    N_LAYERS = 2  # 2层RNN，平衡深度和过拟合
    BIDIRECTIONAL = True  # 使用双向RNN提高性能
    DROPOUT_RATE = 0.5  # 适中的dropout率
    LR = 5e-4  # 适中的学习率

    def __init__(self):
        """初始化配置并创建实验目录"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"exp_{self.timestamp}"
        
        # 创建实验特定的目录
        self.experiment_dir = f"./experiments/{self.experiment_name}/"
        self.model_dir = f"{self.experiment_dir}models/"
        self.figure_dir = f"{self.experiment_dir}figures/"
        self.data_dir = f"{self.experiment_dir}data/"
        self.results_dir = f"{self.experiment_dir}results/"
        
        # 创建所有必要的目录
        for dir_path in [self.experiment_dir, self.model_dir, self.figure_dir, 
                        self.data_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 保存实验配置
        self.save_config()
    
    def save_config(self):
        """保存当前实验配置到JSON文件"""
        config_dict = {
            'timestamp': self.timestamp,
            'experiment_name': self.experiment_name,
            'hyperparameters': {
                'BATCH_SIZE': self.BATCH_SIZE,
                'EMBEDDING_DIM': self.EMBEDDING_DIM,
                'HIDDEN_DIM': self.HIDDEN_DIM,
                'N_LAYERS': self.N_LAYERS,
                'BIDIRECTIONAL': self.BIDIRECTIONAL,
                'DROPOUT_RATE': self.DROPOUT_RATE,
                'LR': self.LR,
                'N_EPOCHS': self.N_EPOCHS,
                'MAX_LENGTH': self.MAX_LENGTH,
                'MIN_FREQ': self.MIN_FREQ
            },
            'device': str(self.DEVICE)
        }
        
        with open(f"{self.results_dir}config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)

    @staticmethod
    def set_seed():
        """设置随机种子以确保跨运行的可重现性"""
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed(Config.SEED)
        torch.backends.cudnn.deterministic = True

class DataProcessor:
    """用于处理IMDB数据集的类，包括分词、构建词汇表和数据加载"""
    def __init__(self, config):
        """使用配置初始化并设置数据处理属性"""
        self.config = config
        # TODO: 定义分词器函数
        # 要求：
        # - 基本分词策略：在小写化和去除标点符号后将文本分割成单词（例如，使用re.sub和split）
        # - 还可以尝试高级分词：使用像NLTK（word_tokenize）或spaCy这样的库进行更复杂的分词
        # - 确保分词器能处理IMDB数据集的文本格式（例如，如果存在则移除HTML标签）
        # - 将标记截断到config.MAX_LENGTH以保持一致的序列长度
        # - 比较每种分词器的词汇表大小和模型性能（准确率，F1分数）
        # - 为每个输入文本返回一个标记列表
        def tokenizer(text):
            """基本分词器：小写化、去除HTML标签和标点符号，然后分割成单词"""
            # 移除HTML标签
            text = re.sub(r'<[^>]+>', '', text)
            # 小写化
            text = text.lower()
            # 去除标点符号，保留字母和数字
            text = re.sub(r'[^\w\s]', ' ', text)
            # 分割成单词并过滤空字符串
            tokens = [token.strip() for token in text.split() if token.strip()]
            return tokens
        
        self.tokenizer = tokenizer
        
        self.vocab = None  
        self.vocab_list = None  
        self.unk_index = None  
        self.pad_index = None  
        self.train_data = None 
        self.valid_data = None  
        self.test_data = None  
        self.train_loader = None  
        self.valid_loader = None  
        self.test_loader = None  

    def load_dataset(self):
        """从Hugging Face加载IMDB数据集"""
        os.makedirs(self.config.data_dir, exist_ok=True)
        train_data, test_data = datasets.load_dataset(
            "imdb", split=["train", "test"], cache_dir=self.config.data_dir)
        return train_data, test_data

    def tokenize_example(self, example):
        """对单个示例进行分词并返回带长度的标记"""
        # TODO: 实现单个示例的分词
        # 要求：
        # - 使用self.tokenizer处理输入的example["text"]
        # - 将结果标记截断到self.config.MAX_LENGTH
        # - 返回一个包含"tokens"（标记列表）和"length"（标记数量）的字典
        # - 确保与IMDB数据集的文本格式兼容
        tokens = self.tokenizer(example["text"])
        # 截断到最大长度
        tokens = tokens[:self.config.MAX_LENGTH]
        return {"tokens": tokens, "length": len(tokens)}

    def build_vocab(self, token_iterator):
        """从已分词的训练数据构建词汇表"""
        counter = Counter()
        for tokens in token_iterator:
            counter.update(tokens)
        vocab = {word for word, freq in counter.items() if freq >= self.config.MIN_FREQ}
        vocab = self.config.VOCAB_SPECIALS + list(vocab)
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        return word_to_idx, vocab

    def numericalize_example(self, example):
        """使用词汇表将标记转换为索引"""
        ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in example["tokens"]]
        return {"ids": ids}

    def get_collate_fn(self):
        """创建用于批处理的整理函数"""
        def collate_fn(batch):
            batch_ids = [item["ids"].clone().detach() for item in batch]
            batch_ids = nn.utils.rnn.pad_sequence(
                batch_ids, padding_value=self.pad_index, batch_first=True
            )
            batch_length = torch.stack([item["length"].clone().detach() for item in batch])
            batch_label = torch.stack([item["label"].clone().detach() for item in batch])
            return {"ids": batch_ids, "length": batch_length, "label": batch_label}
        return collate_fn

    def get_data_loader(self, dataset, shuffle=False):
        """为数据集创建数据加载器"""
        collate_fn = self.get_collate_fn()
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=shuffle
        )

    def process(self):
        """执行完整的数据处理流程"""
        train_data, test_data = self.load_dataset()

        train_data = train_data.map(self.tokenize_example)
        test_data = test_data.map(self.tokenize_example)

        token_iterator = (item["tokens"] for item in train_data)
        self.vocab, self.vocab_list = self.build_vocab(token_iterator)
        self.unk_index = self.vocab["<unk>"]
        self.pad_index = self.vocab["<pad>"]

        train_valid_data = train_data.train_test_split(test_size=self.config.TEST_SIZE)
        self.train_data = train_valid_data["train"]
        self.valid_data = train_valid_data["test"]
        self.test_data = test_data

        self.train_data = self.train_data.map(self.numericalize_example)
        self.valid_data = self.valid_data.map(self.numericalize_example)
        self.test_data = self.test_data.map(self.numericalize_example)

        self.train_data = self.train_data.with_format(type="torch", columns=["ids", "label", "length"])
        self.valid_data = self.valid_data.with_format(type="torch", columns=["ids", "label", "length"])
        self.test_data = self.test_data.with_format(type="torch", columns=["ids", "label", "length"])

        self.train_loader = self.get_data_loader(self.train_data, shuffle=True)
        self.valid_loader = self.get_data_loader(self.valid_data)
        self.test_loader = self.get_data_loader(self.test_data)

class RNNModel(nn.Module):
    """用于情感分析的RNN模型"""
    def __init__(
        self,
        model_type,  
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        # TODO: 实现RNN模型架构
        # 要求：
        # - 支持三种模型类型：'RNN'、'LSTM'、'GRU'，使用PyTorch的nn.RNN、nn.LSTM或nn.GRU
        # - 包含一个嵌入层（nn.Embedding），使用vocab_size和embedding_dim，使用pad_index进行填充
        # - 配置RNN层，包括hidden_dim、n_layers、bidirectional和dropout_rate（batch_first=True）
        # - 添加一个全连接层（nn.Linear）将最终隐藏状态映射到output_dim
        # - 对嵌入和隐藏状态输出应用dropout（nn.Dropout）
        # - 对于双向RNN，连接来自两个方向的最终隐藏状态
        # - 使用nn.utils.rnn.pack_padded_sequence和pad_packed_sequence处理可变长度序列
        # - 通过Config超参数实验不同配置，并比较性能（准确率，F1分数）
        
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        
        # 计算RNN的输入和输出维度
        rnn_input_dim = embedding_dim
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 选择RNN类型
        if model_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout_rate if n_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout_rate if n_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout_rate if n_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接输出层
        self.fc = nn.Linear(rnn_output_dim, output_dim)

    def forward(self, ids, length):
        """定义RNN模型的前向传播"""
        # TODO: 实现前向传播
        # 要求：
        # - 使用嵌入层嵌入输入ids
        # - 对嵌入的输入应用dropout
        # - 打包嵌入序列以处理可变长度
        # - 将打包的序列传递给RNN层（如适用，处理LSTM的单元状态）
        # - 解包RNN输出
        # - 提取最终隐藏状态（如需要，连接双向状态）
        # - 对隐藏状态应用dropout
        # - 将隐藏状态传递给全连接层以获得预测
        # - 返回形状为(batch_size, output_dim)的预测张量
        
        # 使用嵌入层嵌入输入ids
        embedded = self.embedding(ids)
        
        # 对嵌入的输入应用dropout
        embedded = self.dropout(embedded)
        
        # 打包嵌入序列以处理可变长度
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # 将打包的序列传递给RNN层
        if self.model_type == 'LSTM':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)
        
        # 解包RNN输出
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 提取最终隐藏状态
        if self.bidirectional:
            # 对于双向RNN，连接来自两个方向的最终隐藏状态
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # 对于单向RNN，取最后一层的隐藏状态
            hidden = hidden[-1, :, :]
        
        # 对隐藏状态应用dropout
        hidden = self.dropout(hidden)
        
        # 将隐藏状态传递给全连接层以获得预测
        prediction = self.fc(hidden)
        
        # 返回形状为(batch_size, output_dim)的预测张量
        return prediction

class Trainer:
    """用于训练、评估和使用RNN模型进行预测的类"""
    def __init__(self, config, data_processor):
        """使用配置和数据处理器初始化"""
        self.config = config
        self.data_processor = data_processor
        self.models = {
            "RNN": RNNModel("RNN", len(self.data_processor.vocab), self.config.EMBEDDING_DIM, self.config.HIDDEN_DIM,
                            self.config.OUTPUT_DIM, self.config.N_LAYERS, self.config.BIDIRECTIONAL,
                            self.config.DROPOUT_RATE, self.data_processor.pad_index),
            "LSTM": RNNModel("LSTM", len(self.data_processor.vocab), self.config.EMBEDDING_DIM, self.config.HIDDEN_DIM,
                             self.config.OUTPUT_DIM, self.config.N_LAYERS, self.config.BIDIRECTIONAL,
                             self.config.DROPOUT_RATE, self.data_processor.pad_index),
            "GRU": RNNModel("GRU", len(self.data_processor.vocab), self.config.EMBEDDING_DIM, self.config.HIDDEN_DIM,
                            self.config.OUTPUT_DIM, self.config.N_LAYERS, self.config.BIDIRECTIONAL,
                            self.config.DROPOUT_RATE, self.data_processor.pad_index)
        }
        self.criterion = nn.CrossEntropyLoss().to(self.config.DEVICE)
        self.metrics = {name: collections.defaultdict(list) for name in self.models}
        
        # 初始化详细结果存储
        self.detailed_results = {name: {
            'train_losses': [],
            'train_accs': [],
            'valid_losses': [],
            'valid_accs': [],
            'test_results': {},
            'best_epoch': 0,
            'best_valid_loss': float('inf')
        } for name in self.models}

    def initialize_weights(self, m):
        """初始化线性层、RNN和嵌入层的模型权重"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.orthogonal_(param)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.1)

    def get_accuracy(self, prediction, label):
        """计算批次准确率"""
        batch_size, _ = prediction.shape
        predicted_classes = prediction.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(label).sum()
        return correct_predictions / batch_size

    def train(self, model, dataloader, optimizer, model_name, epoch):
        """训练模型一个轮次"""
        model.train()
        epoch_losses = []
        epoch_accs = []
        for batch in tqdm.tqdm(dataloader, desc=f"training on epoch {epoch}..."):
            ids = batch["ids"].to(self.config.DEVICE)
            length = batch["length"]
            label = batch["label"].to(self.config.DEVICE)
            prediction = model(ids, length)
            loss = self.criterion(prediction, label)
            accuracy = self.get_accuracy(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
        return np.mean(epoch_losses), np.mean(epoch_accs)

    def evaluate(self, model, dataloader):
        """在数据集上评估模型"""
        model.eval()
        epoch_losses = []
        epoch_accs = []
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
                ids = batch["ids"].to(self.config.DEVICE)
                length = batch["length"]
                label = batch["label"].to(self.config.DEVICE)
                prediction = model(ids, length)
                loss = self.criterion(prediction, label)
                accuracy = self.get_accuracy(prediction, label)
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())
                predicted_classes = prediction.argmax(dim=-1).cpu().numpy()
                predictions.extend(predicted_classes)
                true_labels.extend(label.cpu().numpy())
        return np.mean(epoch_losses), np.mean(epoch_accs), predictions, true_labels

    def train_and_evaluate(self):
        """训练和评估所有模型，根据验证损失保存最佳模型"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        for model_name, model in self.models.items():
            model = model.to(self.config.DEVICE)
            model.apply(self.initialize_weights)
            optimizer = optim.Adam(model.parameters(), lr=self.config.LR)
            best_valid_loss = float("inf")
            for epoch in range(self.config.N_EPOCHS):
                train_loss, train_acc = self.train(
                    model, self.data_processor.train_loader, optimizer, model_name, epoch
                )
                valid_loss, valid_acc, _, _ = self.evaluate(model, self.data_processor.valid_loader)
                self.metrics[model_name]["train_losses"].append(train_loss)
                self.metrics[model_name]["train_accs"].append(train_acc)
                self.metrics[model_name]["valid_losses"].append(valid_loss)
                self.metrics[model_name]["valid_accs"].append(valid_acc)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), f"{self.config.model_dir}/{model_name}.pt")
                self.detailed_results[model_name]['train_losses'].append(train_loss)
                self.detailed_results[model_name]['train_accs'].append(train_acc)
                self.detailed_results[model_name]['valid_losses'].append(valid_loss)
                self.detailed_results[model_name]['valid_accs'].append(valid_acc)
                self.detailed_results[model_name]['best_epoch'] = epoch
                self.detailed_results[model_name]['best_valid_loss'] = best_valid_loss
                print(f"Model: {model_name}, Epoch: {epoch}")
                print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
                print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
            
            # 保存每个模型的详细训练结果
            self.save_model_results(model_name)

    def visualize_metrics(self):
        """可视化训练和验证指标"""
        os.makedirs(self.config.figure_dir, exist_ok=True)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        for model_name in self.models:
            ax.plot(self.metrics[model_name]["train_losses"], label=f"{model_name} train loss")
            ax.plot(self.metrics[model_name]["valid_losses"], label=f"{model_name} valid loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_xticks(range(self.config.N_EPOCHS))
        ax.legend()
        ax.grid()
        plt.savefig(f"{self.config.figure_dir}/train_valid_loss.png")
        plt.close()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        for model_name in self.models:
            ax.plot(self.metrics[model_name]["train_accs"], label=f"{model_name} train accuracy")
            ax.plot(self.metrics[model_name]["valid_accs"], label=f"{model_name} valid accuracy")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_xticks(range(self.config.N_EPOCHS))
        ax.legend()
        ax.grid()
        plt.savefig(f"{self.config.figure_dir}/train_valid_accuracy.png")
        plt.close()

    def test(self):
        """在测试集上评估模型"""
        print("\nTesting on the full test set:")
        for model_name, model in self.models.items():
            model.load_state_dict(torch.load(f"{self.config.model_dir}/{model_name}.pt"))
            test_loss, test_acc, predictions, true_labels = self.evaluate(
                model, self.data_processor.test_loader
            )
            test_f1 = f1_score(true_labels, predictions, average='weighted')
            print(f"Model: {model_name}")
            print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}, Test F1 Score: {test_f1:.3f}")
            self.detailed_results[model_name]['test_results'] = {
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_f1': test_f1
            }
        
        # 保存所有测试结果
        self.save_test_results()

    def save_model_results(self, model_name):
        """保存单个模型的详细训练结果"""
        results = self.detailed_results[model_name]
        results_file = f"{self.config.results_dir}{model_name}_training_results.json"
        
        # 添加模型配置信息
        model_config = {
            'model_type': model_name,
            'hyperparameters': {
                'embedding_dim': self.config.EMBEDDING_DIM,
                'hidden_dim': self.config.HIDDEN_DIM,
                'n_layers': self.config.N_LAYERS,
                'bidirectional': self.config.BIDIRECTIONAL,
                'dropout_rate': self.config.DROPOUT_RATE,
                'lr': self.config.LR,
                'batch_size': self.config.BATCH_SIZE
            },
            'training_results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(model_config, f, indent=4)
        
        # 保存训练曲线数据为CSV格式（便于Excel分析）
        import pandas as pd
        df = pd.DataFrame({
            'epoch': range(len(results['train_losses'])),
            'train_loss': results['train_losses'],
            'train_acc': results['train_accs'],
            'valid_loss': results['valid_losses'],
            'valid_acc': results['valid_accs']
        })
        csv_file = f"{self.config.results_dir}{model_name}_training_curves.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Saved detailed results for {model_name} to {results_file}")
        print(f"Saved training curves for {model_name} to {csv_file}")

    def save_test_results(self):
        """保存所有模型的测试结果比较"""
        test_results = {}
        for model_name in self.models:
            test_results[model_name] = self.detailed_results[model_name]['test_results']
        
        # 保存测试结果
        test_file = f"{self.config.results_dir}test_results_comparison.json"
        with open(test_file, 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # 创建测试结果比较表格
        import pandas as pd
        comparison_data = []
        for model_name, results in test_results.items():
            comparison_data.append({
                'Model': model_name,
                'Test Loss': f"{results['test_loss']:.4f}",
                'Test Accuracy': f"{results['test_acc']:.4f}",
                'Test F1 Score': f"{results['test_f1']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        csv_file = f"{self.config.results_dir}test_results_comparison.csv"
        df.to_csv(csv_file, index=False)
        
        # 保存最佳模型信息
        best_model = min(test_results.items(), key=lambda x: x[1]['test_loss'])
        best_info = {
            'best_model': best_model[0],
            'best_test_loss': best_model[1]['test_loss'],
            'best_test_acc': best_model[1]['test_acc'],
            'best_test_f1': best_model[1]['test_f1']
        }
        
        best_file = f"{self.config.results_dir}best_model_info.json"
        with open(best_file, 'w') as f:
            json.dump(best_info, f, indent=4)
        
        print(f"Saved test results comparison to {test_file}")
        print(f"Saved test results table to {csv_file}")
        print(f"Saved best model info to {best_file}")
        print(f"\nBest Model: {best_model[0]}")
        print(f"Best Test Loss: {best_model[1]['test_loss']:.4f}")
        print(f"Best Test Accuracy: {best_model[1]['test_acc']:.4f}")
        print(f"Best Test F1 Score: {best_model[1]['test_f1']:.4f}")

    def save_experiment_summary(self):
        """保存实验总结报告"""
        summary = {
            'experiment_name': self.config.experiment_name,
            'timestamp': self.config.timestamp,
            'device': str(self.config.DEVICE),
            'vocab_size': len(self.data_processor.vocab),
            'train_samples': len(self.data_processor.train_data),
            'valid_samples': len(self.data_processor.valid_data),
            'test_samples': len(self.data_processor.test_data),
            'model_comparison': {}
        }
        
        for model_name in self.models:
            model_results = self.detailed_results[model_name]
            summary['model_comparison'][model_name] = {
                'best_epoch': model_results['best_epoch'],
                'best_valid_loss': model_results['best_valid_loss'],
                'final_train_loss': model_results['train_losses'][-1],
                'final_train_acc': model_results['train_accs'][-1],
                'final_valid_loss': model_results['valid_losses'][-1],
                'final_valid_acc': model_results['valid_accs'][-1],
                'test_loss': model_results['test_results']['test_loss'],
                'test_acc': model_results['test_results']['test_acc'],
                'test_f1': model_results['test_results']['test_f1']
            }
        
        summary_file = f"{self.config.results_dir}experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Saved experiment summary to {summary_file}")

    def predict_sentiment(self, text, model_name, max_length=256):
        """预测单个文本的情感"""
        model = self.models[model_name]
        model.eval()
        tokens = self.data_processor.tokenizer(text)[:max_length]
        ids = [self.data_processor.vocab.get(token, self.data_processor.vocab["<unk>"]) for token in tokens]
        length = torch.LongTensor([len(ids)])
        tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(self.config.DEVICE)
        with torch.no_grad():
            prediction = model(tensor, length).squeeze(dim=0)
        probability = torch.softmax(prediction, dim=-1)
        predicted_class = prediction.argmax(dim=-1).item()
        predicted_probability = probability[predicted_class].item()
        return predicted_class, predicted_probability

    def predict_example_texts(self):
        """预测示例文本的情感作为参考"""
        test_texts = [
            "This film is terrible!",
            "This film is great!",
            "This film is not terrible, it's great!",
            "This film is not great, it's terrible!"
        ]
        print("\nPredictions on example texts (for reference):")
        for model_name in self.models:
            print(f"\nPredictions for {model_name}:")
            for text in test_texts:
                predicted_class, predicted_probability = self.predict_sentiment(text, model_name)
                print(f"Text: {text}")
                print(f"Predicted Class: {predicted_class}, Probability: {predicted_probability:.3f}")

def main():
    """运行实验的主函数"""
    config = Config()
    config.set_seed()

    data_processor = DataProcessor(config)
    data_processor.process()

    trainer = Trainer(config, data_processor)
    trainer.train_and_evaluate()
    trainer.visualize_metrics()
    trainer.test()
    trainer.predict_example_texts()
    
    # 保存实验总结
    trainer.save_experiment_summary()
    
    print(f"\n实验完成！所有结果已保存到: {config.experiment_dir}")
    print(f"实验名称: {config.experiment_name}")
    print(f"时间戳: {config.timestamp}")

def run_hyperparameter_experiments():
    """运行超参数实验，比较不同配置的性能"""
    print("开始超参数实验...")
    
    # 定义要测试的超参数组合
    experiments = [
        {
            'name': 'baseline',
            'BATCH_SIZE': 128,
            'EMBEDDING_DIM': 200,
            'HIDDEN_DIM': 200,
            'N_LAYERS': 2,
            'BIDIRECTIONAL': True,
            'DROPOUT_RATE': 0.5,
            'LR': 5e-4
        },
        {
            'name': 'larger_batch',
            'BATCH_SIZE': 256,
            'EMBEDDING_DIM': 200,
            'HIDDEN_DIM': 200,
            'N_LAYERS': 2,
            'BIDIRECTIONAL': True,
            'DROPOUT_RATE': 0.5,
            'LR': 5e-4
        },
        {
            'name': 'larger_embedding',
            'BATCH_SIZE': 128,
            'EMBEDDING_DIM': 300,
            'HIDDEN_DIM': 200,
            'N_LAYERS': 2,
            'BIDIRECTIONAL': True,
            'DROPOUT_RATE': 0.5,
            'LR': 5e-4
        },
        {
            'name': 'larger_hidden',
            'BATCH_SIZE': 128,
            'EMBEDDING_DIM': 200,
            'HIDDEN_DIM': 300,
            'N_LAYERS': 2,
            'BIDIRECTIONAL': True,
            'DROPOUT_RATE': 0.5,
            'LR': 5e-4
        },
        {
            'name': 'deeper_network',
            'BATCH_SIZE': 128,
            'EMBEDDING_DIM': 200,
            'HIDDEN_DIM': 200,
            'N_LAYERS': 3,
            'BIDIRECTIONAL': True,
            'DROPOUT_RATE': 0.5,
            'LR': 5e-4
        },
        {
            'name': 'unidirectional',
            'BATCH_SIZE': 128,
            'EMBEDDING_DIM': 200,
            'HIDDEN_DIM': 200,
            'N_LAYERS': 2,
            'BIDIRECTIONAL': False,
            'DROPOUT_RATE': 0.5,
            'LR': 5e-4
        },
        {
            'name': 'higher_dropout',
            'BATCH_SIZE': 128,
            'EMBEDDING_DIM': 200,
            'HIDDEN_DIM': 200,
            'N_LAYERS': 2,
            'BIDIRECTIONAL': True,
            'DROPOUT_RATE': 0.7,
            'LR': 5e-4
        },
        {
            'name': 'lower_lr',
            'BATCH_SIZE': 128,
            'EMBEDDING_DIM': 200,
            'HIDDEN_DIM': 200,
            'N_LAYERS': 2,
            'BIDIRECTIONAL': True,
            'DROPOUT_RATE': 0.5,
            'LR': 1e-4
        }
    ]
    
    all_results = []
    
    for i, exp_config in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"实验 {i+1}/{len(experiments)}: {exp_config['name']}")
        print(f"{'='*60}")
        
        # 创建新的配置
        config = Config()
        config.set_seed()
        
        # 更新超参数
        for key, value in exp_config.items():
            if key != 'name':
                setattr(config, key, value)
        
        # 重新保存配置
        config.save_config()
        
        # 运行实验
        data_processor = DataProcessor(config)
        data_processor.process()
        
        trainer = Trainer(config, data_processor)
        trainer.train_and_evaluate()
        trainer.visualize_metrics()
        trainer.test()
        trainer.save_experiment_summary()
        
        # 收集结果
        best_model = None
        best_f1 = 0
        for model_name, results in trainer.detailed_results.items():
            if results['test_results']['test_f1'] > best_f1:
                best_f1 = results['test_results']['test_f1']
                best_model = model_name
        
        exp_result = {
            'experiment_name': exp_config['name'],
            'config': exp_config,
            'best_model': best_model,
            'best_f1': best_f1,
            'all_results': trainer.detailed_results
        }
        all_results.append(exp_result)
        
        print(f"实验 {exp_config['name']} 完成，最佳模型: {best_model}, F1: {best_f1:.4f}")
    
    # 保存所有实验结果比较
    comparison_file = f"./experiments/hyperparameter_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
    
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # 创建比较表格
    import pandas as pd
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Experiment': result['experiment_name'],
            'Batch Size': result['config']['BATCH_SIZE'],
            'Embedding Dim': result['config']['EMBEDDING_DIM'],
            'Hidden Dim': result['config']['HIDDEN_DIM'],
            'Layers': result['config']['N_LAYERS'],
            'Bidirectional': result['config']['BIDIRECTIONAL'],
            'Dropout': result['config']['DROPOUT_RATE'],
            'Learning Rate': result['config']['LR'],
            'Best Model': result['best_model'],
            'Best F1 Score': f"{result['best_f1']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    csv_file = f"./experiments/hyperparameter_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*60}")
    print("所有超参数实验完成！")
    print(f"详细结果保存到: {comparison_file}")
    print(f"比较表格保存到: {csv_file}")
    print(f"{'='*60}")
    
    # 找出最佳配置
    best_exp = max(all_results, key=lambda x: x['best_f1'])
    print(f"\n最佳配置: {best_exp['experiment_name']}")
    print(f"最佳F1分数: {best_exp['best_f1']:.4f}")
    print(f"最佳模型: {best_exp['best_model']}")

if __name__ == "__main__":
    # 运行单个实验
    # main()
    
    # 运行超参数实验
    run_hyperparameter_experiments()
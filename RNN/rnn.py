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
    
    # BATCH_SIZE = 
    # EMBEDDING_DIM = 
    # HIDDEN_DIM = 
    # N_LAYERS = 
    # BIDIRECTIONAL = 
    # DROPOUT_RATE = 
    # LR = 

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
        self.tokenizer = None  # 分词器函数 占位符
        
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
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        train_data, test_data = datasets.load_dataset(
            "imdb", split=["train", "test"], cache_dir=self.config.DATA_DIR)
        return train_data, test_data

    def tokenize_example(self, example):
        """对单个示例进行分词并返回带长度的标记"""
        # TODO: 实现单个示例的分词
        # 要求：
        # - 使用self.tokenizer处理输入的example["text"]
        # - 将结果标记截断到self.config.MAX_LENGTH
        # - 返回一个包含"tokens"（标记列表）和"length"（标记数量）的字典
        # - 确保与IMDB数据集的文本格式兼容
        return {"tokens": [], "length": 0}  

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
        pass  # 占位

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
        return torch.zeros(1, self.output_dim)  # 占位

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
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
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
                    torch.save(model.state_dict(), f"{self.config.MODEL_DIR}/{model_name}.pt")
                print(f"Model: {model_name}, Epoch: {epoch}")
                print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
                print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

    def visualize_metrics(self):
        """可视化训练和验证指标"""
        os.makedirs(self.config.FIGURE_DIR, exist_ok=True)
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
        plt.savefig(f"{self.config.FIGURE_DIR}/train_valid_loss.png")
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
        plt.savefig(f"{self.config.FIGURE_DIR}/train_valid_accuracy.png")
        plt.close()

    def test(self):
        """在测试集上评估模型"""
        print("\nTesting on the full test set:")
        for model_name, model in self.models.items():
            model.load_state_dict(torch.load(f"{self.config.MODEL_DIR}/{model_name}.pt"))
            test_loss, test_acc, predictions, true_labels = self.evaluate(
                model, self.data_processor.test_loader
            )
            test_f1 = f1_score(true_labels, predictions, average='weighted')
            print(f"Model: {model_name}")
            print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}, Test F1 Score: {test_f1:.3f}")

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

if __name__ == "__main__":
    main()
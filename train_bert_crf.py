import os
import sys
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json
import logging
from tqdm import tqdm
from src.models.bert_crf_model import DrugQABertCRF  # 改回使用src/models路径

class DrugQADataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        """
        初始化数据集
        
        Args:
            data_file: 数据文件路径
            tokenizer: BERT 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # 构建标签映射
        self.intent_to_idx = {
            "查询规格": 0,
            "查询剂型": 1,
            "查询生产厂家": 2,
            "查询品牌": 3,
            "查询用法用量": 4
        }
        
        self.entity_to_idx = {
            "O": 0,  # 非实体
            "B-DRUG": 1,  # 药品名开始
            "I-DRUG": 2,  # 药品名中间
            "B-MANUFACTURER": 3,  # 生产厂家开始
            "I-MANUFACTURER": 4,  # 生产厂家中间
            "B-SPECIFICATION": 5,  # 规格开始
            "I-SPECIFICATION": 6,  # 规格中间
            "B-DOSAGE": 7,  # 剂型开始
            "I-DOSAGE": 8,  # 剂型中间
            "B-BRAND": 9,  # 品牌开始
            "I-BRAND": 10  # 品牌中间
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取意图标签
        intent_label = self.intent_to_idx[item['intent']]
        
        # 对文本进行编码
        encoding = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 准备实体标签
        entity_labels = [0] * self.max_length  # 初始化为非实体
        for entity in item['entities']:
            start = entity['start']
            end = entity['end']
            entity_type = entity['type']
            
            # 设置实体标签
            entity_labels[start] = self.entity_to_idx[f"B-{entity_type}"]
            for i in range(start + 1, end):
                entity_labels[i] = self.entity_to_idx[f"I-{entity_type}"]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'intent_label': torch.tensor(intent_label),
            'entity_labels': torch.tensor(entity_labels)
        }

def train_model(train_file, val_file, model_save_dir, num_epochs=10, batch_size=16, learning_rate=2e-5):
    """
    训练模型
    
    Args:
        train_file: 训练数据文件路径
        val_file: 验证数据文件路径
        model_save_dir: 模型保存目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 创建数据集
    train_dataset = DrugQADataset(train_file, tokenizer)
    val_dataset = DrugQADataset(val_file, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = DrugQABertCRF(
        bert_model_name='bert-base-chinese',
        num_intent_labels=len(train_dataset.intent_to_idx),
        num_entity_labels=len(train_dataset.entity_to_idx)
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in progress_bar:
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent_label'].to(device)
            entity_labels = batch['entity_labels'].to(device)
            
            # 前向传播
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                intent_labels=intent_labels,
                entity_labels=entity_labels
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            progress_bar.set_postfix({'loss': train_loss / train_steps})
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_label'].to(device)
                entity_labels = batch['entity_labels'].to(device)
                
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    intent_labels=intent_labels,
                    entity_labels=entity_labels
                )
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pt'))
            
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Average training loss: {train_loss / train_steps:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 训练参数
    train_file = 'data/processed/train.json'
    val_file = 'data/processed/val.json'
    model_save_dir = 'D:\yolov11\drugs\src\models\bert_crf_model.py'  # 修改为正确的模型保存目录
    
    # 开始训练
    train_model(
        train_file=train_file,
        val_file=val_file,
        model_save_dir=model_save_dir,
        num_epochs=10,
        batch_size=16,
        learning_rate=2e-5
    ) 
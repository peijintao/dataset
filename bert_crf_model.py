import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF
import logging

class DrugQABertCRF(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', num_intent_labels=5, num_entity_labels=10):
        """
        初始化 BERT + CRF 模型
        
        Args:
            bert_model_name: BERT 预训练模型名称
            num_intent_labels: 意图分类的标签数量
            num_entity_labels: 实体识别的标签数量
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # 加载 BERT 模型和分词器
        try:
            self.bert = BertModel.from_pretrained(bert_model_name)
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        except Exception as e:
            self.logger.error(f"加载 BERT 模型失败: {str(e)}")
            raise
        
        # 意图分类层
        self.intent_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intent_labels)
        )
        
        # 实体识别层
        self.entity_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_entity_labels)
        )
        
        # CRF 层
        self.crf = CRF(num_entity_labels, batch_first=True)
        
        # 损失函数
        self.intent_criterion = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, intent_labels=None, entity_labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入序列的 token IDs
            attention_mask: 注意力掩码
            intent_labels: 意图标签（训练时使用）
            entity_labels: 实体标签（训练时使用）
            
        Returns:
            如果提供了标签，返回损失值；否则返回预测结果
        """
        # BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # 意图分类
        intent_logits = self.intent_classifier(pooled_output)
        
        # 实体识别
        entity_logits = self.entity_classifier(sequence_output)
        
        if intent_labels is not None and entity_labels is not None:
            # 训练模式
            intent_loss = self.intent_criterion(intent_logits, intent_labels)
            entity_loss = -self.crf(entity_logits, entity_labels, mask=attention_mask.bool())
            total_loss = intent_loss + entity_loss
            return total_loss
        else:
            # 预测模式
            intent_pred = torch.argmax(intent_logits, dim=-1)
            entity_pred = self.crf.decode(entity_logits, mask=attention_mask.bool())
            return {
                'intent': intent_pred,
                'entities': entity_pred
            }
    
    def predict(self, text):
        """
        预测单个问题的意图和实体
        
        Args:
            text: 输入文本
            
        Returns:
            包含意图和实体的字典
        """
        self.eval()
        with torch.no_grad():
            # 对输入文本进行编码
            encoding = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 将输入移到正确的设备上
            input_ids = encoding['input_ids'].to(next(self.parameters()).device)
            attention_mask = encoding['attention_mask'].to(next(self.parameters()).device)
            
            # 获取预测结果
            outputs = self(input_ids, attention_mask)
            
            # 将预测结果转换为可读格式
            intent_idx = outputs['intent'].item()
            entities = outputs['entities'][0]  # 取第一个序列的结果
            
            # 将实体标签映射回文本
            entity_spans = []
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            current_entity = None
            start_idx = None
            
            for i, (token, label) in enumerate(zip(tokens, entities)):
                if label != 0:  # 0 表示非实体
                    if current_entity is None:
                        current_entity = label
                        start_idx = i
                    elif label != current_entity:
                        # 保存之前的实体
                        if start_idx is not None:
                            entity_spans.append({
                                'text': ''.join(tokens[start_idx:i]),
                                'type': current_entity,
                                'start': start_idx,
                                'end': i
                            })
                        current_entity = label
                        start_idx = i
            
            # 保存最后一个实体
            if current_entity is not None and start_idx is not None:
                entity_spans.append({
                    'text': ''.join(tokens[start_idx:]),
                    'type': current_entity,
                    'start': start_idx,
                    'end': len(tokens)
                })
            
            return {
                'intent': intent_idx,
                'entities': entity_spans
            } 
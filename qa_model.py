import torch
import torch.nn as nn
import torch.nn.functional as F
from ahocorasick import Automaton
import re
import ahocorasick
from py2neo import Graph
import json
import os

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.min_seq_len = max(filter_sizes)  # 记录最大的卷积核尺寸
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 处理序列长度小于最大卷积核尺寸的情况
        if x.size(1) < self.min_seq_len:
            # 使用填充将序列长度扩展到最小要求
            padding_len = self.min_seq_len - x.size(1)
            padding = torch.zeros(x.size(0), padding_len, x.size(2)).to(x.device)
            x = torch.cat([x, padding], dim=1)
        
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        
        # 应用卷积和池化
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        # 拼接
        x = torch.cat(x, 1)
        x = self.dropout(x)
        
        # 全连接层
        return self.fc(x)

class DrugQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=[2,3,4], num_classes=4):
        super().__init__()
        
        # 初始化AC自动机
        self.ac = ahocorasick.Automaton()
        
        # 加载同义词和缩写字典
        self.synonyms_dict = {}
        self.abbreviations_dict = {}
        self._load_dictionaries()
        
        # 构建优化后的AC自动机
        self._build_optimized_ac_automaton()
        
        # TextCNN用于文本分类
        self.text_cnn = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            num_classes=num_classes
        )
    
    def _load_dictionaries(self):
        """加载同义词和缩写字典"""
        try:
            # 加载同义词字典
            synonyms_path = os.path.join('data', 'dictionaries', 'synonyms.json')
            if os.path.exists(synonyms_path):
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    self.synonyms_dict = json.load(f)
            
            # 加载缩写字典
            abbreviations_path = os.path.join('data', 'dictionaries', 'abbreviations.json')
            if os.path.exists(abbreviations_path):
                with open(abbreviations_path, 'r', encoding='utf-8') as f:
                    self.abbreviations_dict = json.load(f)
                    
        except Exception as e:
            print(f"加载字典时出错: {str(e)}")
    
    def _build_optimized_ac_automaton(self):
        """构建优化后的AC自动机"""
        try:
            # 连接Neo4j获取所有药品名称
            graph = Graph("neo4j://localhost:7687", auth=("neo4j", "pjt.neo4j.147"))
            query = "MATCH (d:Drug) RETURN d.name as name"
            drug_names = [r['name'] for r in graph.run(query).data()]
            
            # 使用集合去重
            unique_names = set()
            
            # 添加原始药品名称
            for name in drug_names:
                if name:
                    unique_names.add(name)
            
            # 添加同义词
            for name, synonyms in self.synonyms_dict.items():
                unique_names.add(name)
                unique_names.update(synonyms)
            
            # 添加缩写
            for abbr, full_name in self.abbreviations_dict.items():
                unique_names.add(abbr)
                unique_names.add(full_name)
            
            # 按长度降序排序，优先匹配较长的药品名
            sorted_names = sorted(unique_names, key=len, reverse=True)
            
            # 批量添加到AC自动机
            for i, name in enumerate(sorted_names):
                self.ac.add_word(name, (i, name))
            
            # 构建自动机
            self.ac.make_automaton()
            print(f"优化后的AC自动机初始化完成，加载了 {len(sorted_names)} 个药品名称")
            
        except Exception as e:
            print(f"构建AC自动机时出错: {str(e)}")
            # 添加一些默认药品名称作为备份
            default_names = ['阿司匹林', '布洛芬', '感冒灵']
            for i, name in enumerate(default_names):
                self.ac.add_word(name, (i, name))
            self.ac.make_automaton()
    
    def _resolve_synonyms(self, drug_name):
        """解析药品同义词"""
        # 检查是否是同义词
        for main_name, synonyms in self.synonyms_dict.items():
            if drug_name in synonyms:
                return main_name
        return drug_name
    
    def _resolve_abbreviations(self, drug_name):
        """解析药品缩写"""
        return self.abbreviations_dict.get(drug_name, drug_name)
    
    def _disambiguate_drug_name(self, drug_name, context):
        """基于上下文进行药品名称消歧"""
        # 如果药品名称在同义词字典中
        if drug_name in self.synonyms_dict:
            # 检查上下文中是否包含其他同义词
            for synonym in self.synonyms_dict[drug_name]:
                if synonym in context:
                    return synonym
            return drug_name
        
        # 如果药品名称在缩写字典中
        if drug_name in self.abbreviations_dict:
            # 检查上下文中是否包含完整名称
            full_name = self.abbreviations_dict[drug_name]
            if full_name in context:
                return full_name
            return drug_name
        
        return drug_name
    
    def find_drug_name(self, text):
        """使用优化后的AC自动机提取药品名称"""
        try:
            # 使用AC自动机查找所有匹配
            matches = []
            for end_index, (_, original_value) in self.ac.iter(text):
                start_index = end_index - len(original_value) + 1
                matches.append((start_index, end_index, original_value))
            
            if matches:
                # 按匹配长度和位置排序
                matches.sort(key=lambda x: (-len(x[2]), x[0]))
                
                # 获取最佳匹配
                best_match = matches[0][2]
                
                # 解析同义词和缩写
                resolved_name = self._resolve_synonyms(best_match)
                resolved_name = self._resolve_abbreviations(resolved_name)
                
                # 基于上下文进行消歧
                final_name = self._disambiguate_drug_name(resolved_name, text)
                
                return final_name
            
            # 如果没有匹配，使用正则表达式作为备份
            patterns = [
                r'(.+?)的规格',
                r'(.+?)的剂型',
                r'(.+?)是什么剂型',
                r'(.+?)的生产厂家',
                r'谁生产的(.+?)[?？]',
                r'(.+?)是谁生产的',
                r'(.+?)的品牌',
                r'(.+?)是什么牌子',
                r'(.+?)的厂家'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    drug_name = match.group(1).strip()
                    if len(drug_name) <= 20:  # 限制长度避免错误匹配
                        # 解析同义词和缩写
                        resolved_name = self._resolve_synonyms(drug_name)
                        resolved_name = self._resolve_abbreviations(resolved_name)
                        return self._disambiguate_drug_name(resolved_name, text)
            
            # 如果还是没找到，返回问题的前几个字符
            words = text.split('的')[0]
            if '是' in words:
                words = words.split('是')[0]
            return words.strip()[:10]
            
        except Exception as e:
            print(f"药品名称提取出错: {str(e)}")
            return text.split('的')[0][:10]  # 返回基本的提取结果
    
    def forward(self, text_ids):
        # 使用TextCNN进行分类
        return self.text_cnn(text_ids)
    
    @property
    def type_to_idx(self):
        return {
            'specifications': 0,
            'dosage_form': 1,
            'manufacturer': 2,
            'drug_brand': 3
        } 
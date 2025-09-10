from neo4j import GraphDatabase
import mysql.connector
import os
from typing import List, Dict
from tqdm import tqdm


class SQLToNeo4jConverter:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        # 只初始化Neo4j连接
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )

    def close(self):
        """关闭数据库连接"""
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def clear_neo4j_database(self):
        """清空Neo4j数据库"""
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Neo4j数据库已清空")

    def create_neo4j_schema(self):
        """创建Neo4j数据库模式和索引"""
        with self.neo4j_driver.session() as session:
            # 创建唯一性约束
            session.run("""
                CREATE CONSTRAINT drug_name IF NOT EXISTS
                FOR (d:Drug) REQUIRE d.name IS UNIQUE
            """)
            print("创建药品名称唯一性约束")

            # 创建索引
            session.run("""
                CREATE INDEX drug_name_idx IF NOT EXISTS
                FOR (d:Drug) ON (d.name)
            """)
            print("创建药品名称索引")

    def import_from_sql_file(self, sql_file: str):
        """从SQL文件导入数据"""
        try:
            # 检查文件是否存在
            if not os.path.exists(sql_file):
                raise FileNotFoundError(f"SQL文件不存在: {sql_file}")
            
            # 读取SQL文件
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 解析INSERT语句中的数据
            drugs_data = []
            # 分割多个语句
            statements = sql_content.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if statement.startswith('INSERT INTO `db_drugs`'):
                    # 提取VALUES部分
                    values_start = statement.find('VALUES')
                    if values_start != -1:
                        values_part = statement[values_start + 6:].strip()
                        # 解析每组值
                        value_groups = values_part.split('),(')
                        for group in value_groups:
                            # 清理括号
                            group = group.strip('()')
                            # 分割字段
                            fields = self._split_fields(group)
                            if len(fields) >= 13:  # 确保有足够的字段
                                drug = {
                                    'drug_id': fields[0],
                                    'bar_code': fields[1],
                                    'drug_name': fields[2],
                                    'specifications': fields[3],
                                    'dosage_form': fields[4],
                                    'approval_number': fields[5],
                                    'manufacturer': fields[6],
                                    'drug_type': fields[7],
                                    'drug_brand': fields[8],
                                    'indication': fields[9],
                                    'usage': fields[10],
                                    'adverse_reactions': fields[11],
                                    'contraindication': fields[12]
                                }
                                drugs_data.append(drug)
            
            if not drugs_data:
                # 打印文件内容的一部分，帮助调试
                print("SQL文件内容预览:")
                print(sql_content[:500])
                raise ValueError("未从SQL文件中解析到任何药品数据")
            
            # 清空现有数据库
            print("清空现有数据库...")
            self.clear_neo4j_database()
            
            # 创建数据库模式
            print("创建数据库模式...")
            self.create_neo4j_schema()
            
            # 导入数据到Neo4j
            print(f"解析到 {len(drugs_data)} 条药品数据")
            with tqdm(total=len(drugs_data), desc="导入数据") as pbar:
                with self.neo4j_driver.session() as session:
                    for drug in drugs_data:
                        self._create_drug_nodes(session, drug)
                        pbar.update(1)
            
            print("\n数据导入完成！")
            print(f"总共导入 {len(drugs_data)} 条药品数据")
            
        except Exception as e:
            print(f"导入过程中出现错误: {str(e)}")
            raise

    def _split_fields(self, value_group: str) -> List[str]:
        """分割字段值"""
        fields = []
        current_field = ''
        in_quotes = False
        
        for char in value_group:
            if char == "'" and (len(current_field) == 0 or current_field[-1] != '\\'):
                in_quotes = not in_quotes
                current_field += char
            elif char == ',' and not in_quotes:
                fields.append(self._clean_field(current_field))
                current_field = ''
            else:
                current_field += char
        
        # 添加最后一个字段
        if current_field:
            fields.append(self._clean_field(current_field))
        
        return fields

    def _clean_field(self, field: str) -> str:
        """清理字段值"""
        field = field.strip()
        # 移除引号
        if field.startswith("'") and field.endswith("'"):
            field = field[1:-1]
        # 处理NULL值
        if field.lower() == 'null':
            field = ''
        # 清理转义字符
        field = field.replace("\\'", "'")
        return field

    def _create_drug_nodes(self, session, drug: Dict):
        """创建药品相关的节点和关系"""
        # 创建药品节点
        session.run("""
            MERGE (d:Drug {name: $drug_name})
            SET d.specifications = $specifications,
                d.dosage_form = $dosage_form,
                d.approval_number = $approval_number,
                d.drug_type = $drug_type,
                d.drug_brand = $drug_brand,
                d.manufacturer = $manufacturer
        """, {
            'drug_name': drug['drug_name'],
            'specifications': drug['specifications'],
            'dosage_form': drug['dosage_form'],
            'approval_number': drug['approval_number'],
            'drug_type': drug['drug_type'],
            'drug_brand': drug['drug_brand'],
            'manufacturer': drug['manufacturer']
        })
        
        # 创建适应症(功效)节点和关系
        if drug.get('indication'):
            for effect in drug['indication'].split('。'):
                if effect.strip():
                    session.run("""
                        MATCH (d:Drug {name: $drug_name})
                        MERGE (e:Effect {description: $effect})
                        MERGE (d)-[:HAS_EFFECT]->(e)
                    """, drug_name=drug['drug_name'], effect=effect.strip())
        
        # 创建用法用量节点和关系
        if drug.get('usage'):
            for usage in drug['usage'].split('。'):
                if usage.strip():
                    session.run("""
                        MATCH (d:Drug {name: $drug_name})
                        MERGE (u:Usage {description: $usage})
                        MERGE (d)-[:HAS_USAGE]->(u)
                    """, drug_name=drug['drug_name'], usage=usage.strip())
        
        # 创建不良反应节点和关系
        if drug.get('adverse_reactions'):
            for reaction in drug['adverse_reactions'].split('。'):
                if reaction.strip():
                    session.run("""
                        MATCH (d:Drug {name: $drug_name})
                        MERGE (s:SideEffect {description: $reaction})
                        MERGE (d)-[:HAS_SIDE_EFFECT]->(s)
                    """, drug_name=drug['drug_name'], reaction=reaction.strip())
        
        # 创建禁忌节点和关系
        if drug.get('contraindication'):
            for contra in drug['contraindication'].split('。'):
                if contra.strip():
                    session.run("""
                        MATCH (d:Drug {name: $drug_name})
                        MERGE (c:Contraindication {description: $contra})
                        MERGE (d)-[:HAS_CONTRAINDICATION]->(c)
                    """, drug_name=drug['drug_name'], contra=contra.strip())


def main():
    # 修改为您的Neo4j配置
    neo4j_config = {
        'uri': "neo4j://localhost:7687",  # 使用bolt协议
        'user': "neo4j",
        'password': "pjt.neo4j.147"
    }

    # 使用清理后的SQL文件
    try:
        # 创建转换器实例
        converter = SQLToNeo4jConverter(
            neo4j_uri=neo4j_config['uri'],
            neo4j_user=neo4j_config['user'],
            neo4j_password=neo4j_config['password']
        )

        # 从清理后的SQL文件读取数据并导入
        print("开始导入数据...")
        # 修改为清理后的SQL文件路径
        converter.import_from_sql_file('D:/yolov11/drugs/db_drugs_cleaned.sql')

        print("数据导入完成！")

    except Exception as e:
        print(f"发生错误: {str(e)}")

    finally:
        if 'converter' in locals():
            converter.close()


if __name__ == "__main__":
    main() 
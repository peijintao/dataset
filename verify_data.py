from neo4j import GraphDatabase

def verify_data():
    driver = GraphDatabase.driver(
        "neo4j://localhost:7687",
        auth=("neo4j", "pjt.neo4j.147")
    )
    
    try:
        with driver.session() as session:
            # 检查基本药品数据
            result = session.run("""
                MATCH (d:Drug)
                RETURN count(d) as drug_count,
                       count(d.specifications) as spec_count,
                       count(d.manufacturer) as mfr_count,
                       count(d.dosage_form) as form_count,
                       count(d.drug_brand) as brand_count
            """)
            stats = result.single()
            print("\n基本数据统计:")
            print(f"药品总数: {stats['drug_count']}")
            print(f"有规格信息的药品数: {stats['spec_count']}")
            print(f"有厂家信息的药品数: {stats['mfr_count']}")
            print(f"有剂型信息的药品数: {stats['form_count']}")
            print(f"有品牌信息的药品数: {stats['brand_count']}")
            
            # 检查时序数据
            result = session.run("""
                MATCH ()-[r:HAS_SPEC_HISTORY]->() RETURN count(r) as spec_history
            """)
            spec_history = result.single()['spec_history']
            
            result = session.run("""
                MATCH ()-[r:HAS_MANUFACTURER_HISTORY]->() RETURN count(r) as mfr_history
            """)
            mfr_history = result.single()['mfr_history']
            
            print("\n时序数据统计:")
            print(f"规格变更历史记录数: {spec_history}")
            print(f"厂家变更历史记录数: {mfr_history}")
            
            # 显示一些示例数据
            print("\n示例数据:")
            result = session.run("""
                MATCH (d:Drug)-[:HAS_SPEC_HISTORY]->(h)
                RETURN d.name as name, h.value as spec, h.start_time as start_time
                LIMIT 3
            """)
            for record in result:
                print(f"\n药品: {record['name']}")
                print(f"规格: {record['spec']}")
                print(f"开始时间: {record['start_time']}")
                
    finally:
        driver.close()

if __name__ == "__main__":
    verify_data() 
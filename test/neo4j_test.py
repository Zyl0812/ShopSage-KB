from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "123456789"))
driver.verify_connectivity()
print('连接成功')


with driver.session(database='neo4j') as session:
    # 清空
    session.run('MATCH (n) DETACH DELETE n').consume()
    
    # 创建节点
    session.run(
        'CREATE (:Customer {name: $name, age: $age, vip: $vip})',
        name='张三', age=28, vip=True
    ).consume()
    
    result = session.run(
        'MATCH (n) RETURN n LIMIT 25;'
    )
    for record in result:
        print(record.data())
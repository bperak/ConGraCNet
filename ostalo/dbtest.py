from neo4j import GraphDatabase

uri = "bolt://polinom.uniri.hr:7687"  # Or neo4j:// for routing
driver = GraphDatabase.driver(uri, auth=("neo4j", "neo4j"))

with driver.session() as session:
    result = session.run("MATCH (n) RETURN n LIMIT 5")
    for record in result:
        print(record)

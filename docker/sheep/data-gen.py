from kafka import KafkaProducer
from faker import Faker
from kafka.errors import NoBrokersAvailable
from datetime import datetime
import pandas as pd
import json
import time
import os 

fake = Faker()
#get the kafka listner from the env variables li definit f'docker compose 
BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS')



def create_kafka_producer():
    max_retries=10
    retry=0
    while max_retries>retry:
        try:
            producer=KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("connected to Kafka successfully")
            return producer
        except NoBrokersAvailable:
            print("Kafka container is yet not available")
            retry+=1
            time.sleep(5)


#-------------------------------------Start here---------------------------
producer=create_kafka_producer()
while True:
    df=pd.read_csv("sheep_gps_data.csv")

    for _, row in df.iterrows():
        message=row.to_dict()
        producer.send("sheep-topic",value=message)
        print(f"produce data {message}")
        time.sleep(0.5)

print("All data has been produced successfully")
producer.flush()
producer.close()

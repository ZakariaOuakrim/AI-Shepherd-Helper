#!/usr/bin/env python3
import json
import time
import numpy as np
from kafka import KafkaConsumer

def distance_m(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

print("🐑 Starting Robust Sheep Monitor...")

try:
    # Create consumer with manual deserialization to handle errors
    consumer = KafkaConsumer(
        "sheep-topic",
        bootstrap_servers=[
            "kafka-0.kafka-headless.kafka.svc.cluster.local:9092",
            "kafka-1.kafka-headless.kafka.svc.cluster.local:9092"
        ],
        auto_offset_reset="earliest",
        group_id="sheep-monitor",
        value_deserializer=None  # We'll handle JSON manually
    )
    
    print("✅ Connected to Kafka successfully")
    
    sheep_positions = {}
    message_count = 0
    valid_messages = 0
    last_analysis = time.time()
    
    for message in consumer:
        message_count += 1
        
        try:
            # Manual JSON parsing with error handling
            raw_value = message.value
            if raw_value is None or len(raw_value) == 0:
                print(f"⚠️  Message {message_count}: Empty message, skipping")
                continue
            
            # Decode and parse JSON
            json_str = raw_value.decode('utf-8')
            data = json.loads(json_str)
            
            # Validate required fields
            if not all(key in data for key in ['sheep_id', 'latitude', 'longitude']):
                print(f"⚠️  Message {message_count}: Missing required fields, skipping")
                continue
            
            sheep_id = data['sheep_id']
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            
            sheep_positions[sheep_id] = {
                'lat': lat, 
                'lon': lon, 
                'timestamp': data.get('timestamp', 'unknown')
            }
            
            valid_messages += 1
            
            # Print every 20th valid message
            if valid_messages % 20 == 1:
                print(f"📨 Valid message {valid_messages}/{message_count}: {sheep_id} at ({lat:.6f}, {lon:.6f})")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Error in message {message_count}: {e}")
            print(f"   Raw message: {raw_value}")
            continue
        except ValueError as e:
            print(f"❌ Value Error in message {message_count}: {e}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error in message {message_count}: {e}")
            continue
        
        # Analyze every 30 seconds or every 100 valid messages
        current_time = time.time()
        if (current_time - last_analysis >= 30 or valid_messages % 100 == 0) and len(sheep_positions) >= 2:
            last_analysis = current_time
            
            print(f"\n🔍 Analysis time! ({len(sheep_positions)} sheep, {valid_messages} valid messages)")
            
            try:
                sheep_list = list(sheep_positions.items())
                alerts = 0
                
                for i, (sheep_id, pos) in enumerate(sheep_list):
                    distances = []
                    
                    # Calculate distances to all other sheep
                    for j, (other_id, other_pos) in enumerate(sheep_list):
                        if i != j:
                            dist = distance_m(pos['lat'], pos['lon'], other_pos['lat'], other_pos['lon'])
                            distances.append(dist)
                    
                    if distances:
                        min_dist = min(distances)
                        avg_dist = sum(distances) / len(distances)
                        
                        status = "🚨 ALERT" if min_dist > 500 else "✅ OK"
                        print(f"   {status} {sheep_id}: nearest={min_dist:.0f}m, avg={avg_dist:.0f}m")
                        
                        if min_dist > 500:
                            alerts += 1
                            print(f"🚨 ISOLATION ALERT: {sheep_id} is {min_dist:.0f}m from nearest sheep!")
                
                if alerts == 0:
                    print("✅ All sheep are close to the herd")
                else:
                    print(f"🚨 Total alerts: {alerts}")
                    
            except Exception as e:
                print(f"❌ Error during analysis: {e}")
            
            print("-" * 50)

except KeyboardInterrupt:
    print("\n🛑 Stopping consumer...")
except Exception as e:
    print(f"❌ Fatal error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'consumer' in locals():
        consumer.close()
    print("✅ Consumer closed")
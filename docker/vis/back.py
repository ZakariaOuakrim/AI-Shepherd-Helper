#!/usr/bin/env python3
"""
WebSocket server that consumes sheep data from Kafka and broadcasts to web clients
"""

import asyncio
import json
import websockets
import numpy as np
from kafka import KafkaConsumer
from threading import Thread
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SheepWebSocketServer:
    def __init__(self):
        self.clients = set()
        self.sheep_data = {}
        self.alerts = set()
        self.consumer = None
        self.distance_threshold = 500
        
    def distance_m(self, lat1, lon1, lat2, lon2):
        """Calculate distance in meters between two GPS coordinates."""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
        
    def start_kafka_consumer(self):
        """Start consuming from Kafka in a separate thread"""
        def consume():
            try:
                self.consumer = KafkaConsumer(
                    "sheep-topic",
                    bootstrap_servers=[
                        "kafka-0.kafka-headless.kafka.svc.cluster.local:9092",
                        "kafka-1.kafka-headless.kafka.svc.cluster.local:9092"
                    ],
                    auto_offset_reset="latest",  # Only get new messages
                    group_id="websocket-server",
                    value_deserializer=None
                )
                
                logger.info("✅ Connected to Kafka")
                
                for message in self.consumer:
                    try:
                        # Parse message
                        raw_value = message.value
                        if raw_value is None or len(raw_value) == 0:
                            continue
                            
                        json_str = raw_value.decode('utf-8')
                        data = json.loads(json_str)
                        
                        # Validate required fields
                        if not all(key in data for key in ['sheep_id', 'latitude', 'longitude']):
                            continue
                            
                        sheep_id = data['sheep_id']
                        lat = float(data['latitude'])
                        lon = float(data['longitude'])
                        timestamp = data.get('timestamp', datetime.now().isoformat())
                        
                        # Update sheep data
                        self.sheep_data[sheep_id] = {
                            'sheep_id': sheep_id,
                            'latitude': lat,
                            'longitude': lon,
                            'timestamp': timestamp,
                            'last_update': time.time()
                        }
                        
                        # Check for alerts
                        alert = self.check_isolation(sheep_id)
                        
                        # Prepare message for WebSocket clients
                        ws_message = {
                            'sheep_id': sheep_id,
                            'latitude': lat,
                            'longitude': lon,
                            'timestamp': timestamp,
                            'alert': alert
                        }
                        
                        # Broadcast to all connected clients
                        asyncio.run_coroutine_threadsafe(
                            self.broadcast(ws_message), 
                            asyncio.get_event_loop()
                        )
                        
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Kafka consumer error: {e}")
                
        # Start consumer in thread
        consumer_thread = Thread(target=consume, daemon=True)
        consumer_thread.start()
        logger.info("🐑 Kafka consumer started in background")
        
    def check_isolation(self, sheep_id):
        """Check if a sheep is isolated from the herd"""
        if len(self.sheep_data) < 2:
            return False
            
        current_sheep = self.sheep_data[sheep_id]
        current_lat = current_sheep['latitude']
        current_lon = current_sheep['longitude']
        
        min_distance = float('inf')
        
        for other_id, other_data in self.sheep_data.items():
            if other_id == sheep_id:
                continue
                
            distance = self.distance_m(
                current_lat, current_lon,
                other_data['latitude'], other_data['longitude']
            )
            
            min_distance = min(min_distance, distance)
            
        is_alert = min_distance > self.distance_threshold
        
        if is_alert:
            self.alerts.add(sheep_id)
            logger.info(f"🚨 ALERT: {sheep_id} is {min_distance:.0f}m from nearest sheep")
        else:
            self.alerts.discard(sheep_id)
            
        return is_alert
        
    async def broadcast(self, message):
        """Broadcast message to all connected clients"""
        if self.clients:
            message_str = json.dumps(message)
            # Send to all clients, remove disconnected ones
            disconnected = set()
            
            for client in self.clients:
                try:
                    await client.send(message_str)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    disconnected.add(client)
                    
            # Remove disconnected clients
            self.clients -= disconnected
            
    async def handle_client(self, websocket, path):
        """Handle new WebSocket client connection"""
        logger.info(f"🌐 New client connected from {websocket.remote_address}")
        self.clients.add(websocket)
        
        try:
            # Send current sheep data to new client
            for sheep_id, data in self.sheep_data.items():
                alert = sheep_id in self.alerts
                message = {
                    'sheep_id': sheep_id,
                    'latitude': data['latitude'],
                    'longitude': data['longitude'],
                    'timestamp': data['timestamp'],
                    'alert': alert
                }
                await websocket.send(json.dumps(message))
                
            # Keep connection alive
            async for message in websocket:
                # Handle any incoming messages from client if needed
                logger.info(f"Received from client: {message}")
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Client disconnected")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            self.clients.discard(websocket)
            
    def cleanup_stale_data(self):
        """Remove sheep that haven't been updated recently"""
        current_time = time.time()
        stale_threshold = 300  # 5 minutes
        
        stale_sheep = [
            sheep_id for sheep_id, data in self.sheep_data.items()
            if current_time - data['last_update'] > stale_threshold
        ]
        
        for sheep_id in stale_sheep:
            del self.sheep_data[sheep_id]
            self.alerts.discard(sheep_id)
            logger.info(f"🧹 Removed stale sheep: {sheep_id}")
            
    async def start_server(self, host="0.0.0.0", port=8080):
        """Start the WebSocket server"""
        logger.info(f"🚀 Starting WebSocket server on {host}:{port}")
        
        # Start Kafka consumer
        self.start_kafka_consumer()
        
        # Start cleanup task
        async def cleanup_task():
            while True:
                await asyncio.sleep(60)  # Run every minute
                self.cleanup_stale_data()
                
        asyncio.create_task(cleanup_task())
        
        # Start WebSocket server
        server = await websockets.serve(self.handle_client, host, port)
        logger.info("✅ WebSocket server started, waiting for clients...")
        
        return server

async def main():
    server_instance = SheepWebSocketServer()
    server = await server_instance.start_server()
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down server...")
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
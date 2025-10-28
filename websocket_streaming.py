"""
WebSocket Real-time Streaming System for Public Health Monitor
Implements real-time data streaming, live alerts, and ML prediction updates
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    logger.warning("WebSockets not available. Install websockets package for real-time streaming.")
    WEBSOCKETS_AVAILABLE = False

from database_service import db_service
from database_models import get_db
from auth import get_current_user_id

class EventType(Enum):
    """Types of real-time events"""
    HEALTH_DATA_UPDATE = "health_data_update"
    ALERT_NOTIFICATION = "alert_notification"
    ML_PREDICTION_UPDATE = "ml_prediction_update"
    SYSTEM_STATUS = "system_status"
    USER_ACTIVITY = "user_activity"
    CONNECTION_STATUS = "connection_status"

@dataclass
class StreamEvent:
    """Real-time stream event"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str]
    location_id: Optional[str]
    data: Dict[str, Any]
    target_users: Optional[List[str]] = None
    priority: str = "normal"  # normal, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'location_id': self.location_id,
            'data': self.data,
            'priority': self.priority
        }

class ConnectionManager:
    """Manages WebSocket connections and user sessions"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}  # connection_id -> websocket
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)  # user_id -> connection_ids
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}  # connection_id -> metadata
        self.location_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # location_id -> connection_ids
        self.event_types_subscribed: Dict[str, Set[EventType]] = defaultdict(set)  # connection_id -> event_types
        
        # Connection statistics
        self.total_connections = 0
        self.active_connections = 0
        self.messages_sent = 0
        self.events_processed = 0
    
    async def connect(self, websocket: WebSocketServerProtocol, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Register a new WebSocket connection"""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available for connection")
            return ""
        
        connection_id = str(uuid.uuid4())
        
        try:
            self.connections[connection_id] = websocket
            self.user_connections[user_id].add(connection_id)
            self.connection_metadata[connection_id] = {
                'user_id': user_id,
                'connected_at': datetime.utcnow(),
                'last_ping': datetime.utcnow(),
                'ip_address': websocket.remote_address[0] if hasattr(websocket, 'remote_address') else 'unknown',
                **(metadata or {})
            }
            
            self.total_connections += 1
            self.active_connections += 1
            
            logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
            
            # Send connection confirmation
            await self.send_to_connection(connection_id, StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.CONNECTION_STATUS,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                location_id=None,
                data={'status': 'connected', 'connection_id': connection_id}
            ))
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
            return ""
    
    async def disconnect(self, connection_id: str):
        """Unregister a WebSocket connection"""
        if connection_id not in self.connections:
            return
        
        try:
            # Get user ID before cleanup
            metadata = self.connection_metadata.get(connection_id, {})
            user_id = metadata.get('user_id')
            
            # Remove from all tracking
            del self.connections[connection_id]
            
            if user_id and connection_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
            
            # Remove from location subscriptions
            for location_id in list(self.location_subscriptions.keys()):
                if connection_id in self.location_subscriptions[location_id]:
                    self.location_subscriptions[location_id].remove(connection_id)
                    if not self.location_subscriptions[location_id]:
                        del self.location_subscriptions[location_id]
            
            # Remove event type subscriptions
            if connection_id in self.event_types_subscribed:
                del self.event_types_subscribed[connection_id]
            
            self.active_connections = max(0, self.active_connections - 1)
            
            logger.info(f"WebSocket disconnected: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket {connection_id}: {e}")
    
    async def subscribe_to_location(self, connection_id: str, location_id: str):
        """Subscribe connection to location-specific events"""
        if connection_id in self.connections:
            self.location_subscriptions[location_id].add(connection_id)
            logger.info(f"Connection {connection_id} subscribed to location {location_id}")
    
    async def subscribe_to_events(self, connection_id: str, event_types: List[EventType]):
        """Subscribe connection to specific event types"""
        if connection_id in self.connections:
            self.event_types_subscribed[connection_id].update(event_types)
            logger.info(f"Connection {connection_id} subscribed to events: {[e.value for e in event_types]}")
    
    async def send_to_connection(self, connection_id: str, event: StreamEvent) -> bool:
        """Send event to a specific connection"""
        if not WEBSOCKETS_AVAILABLE or connection_id not in self.connections:
            return False
        
        try:
            websocket = self.connections[connection_id]
            message = json.dumps(event.to_dict())
            await websocket.send(message)
            
            self.messages_sent += 1
            return True
            
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            # Connection might be broken, clean it up
            await self.disconnect(connection_id)
            return False
    
    async def send_to_user(self, user_id: str, event: StreamEvent) -> int:
        """Send event to all connections for a specific user"""
        sent_count = 0
        
        if user_id in self.user_connections:
            connection_ids = list(self.user_connections[user_id])
            
            for connection_id in connection_ids:
                success = await self.send_to_connection(connection_id, event)
                if success:
                    sent_count += 1
        
        return sent_count
    
    async def broadcast_to_location(self, location_id: str, event: StreamEvent) -> int:
        """Broadcast event to all connections subscribed to a location"""
        sent_count = 0
        
        if location_id in self.location_subscriptions:
            connection_ids = list(self.location_subscriptions[location_id])
            
            for connection_id in connection_ids:
                success = await self.send_to_connection(connection_id, event)
                if success:
                    sent_count += 1
        
        return sent_count
    
    async def broadcast_to_all(self, event: StreamEvent) -> int:
        """Broadcast event to all active connections"""
        sent_count = 0
        
        connection_ids = list(self.connections.keys())
        
        for connection_id in connection_ids:
            success = await self.send_to_connection(connection_id, event)
            if success:
                sent_count += 1
        
        return sent_count
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'total_connections': self.total_connections,
            'active_connections': self.active_connections,
            'messages_sent': self.messages_sent,
            'events_processed': self.events_processed,
            'locations_subscribed': len(self.location_subscriptions),
            'unique_users': len(self.user_connections)
        }

class EventProcessor:
    """Processes and routes real-time events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.event_queue = asyncio.Queue(maxsize=1000)
        self.processing = False
        
        # Event handlers
        self.event_handlers = {
            EventType.HEALTH_DATA_UPDATE: self._handle_health_data_update,
            EventType.ALERT_NOTIFICATION: self._handle_alert_notification,
            EventType.ML_PREDICTION_UPDATE: self._handle_ml_prediction_update,
            EventType.SYSTEM_STATUS: self._handle_system_status,
            EventType.USER_ACTIVITY: self._handle_user_activity
        }
    
    async def start_processing(self):
        """Start the event processing loop"""
        if self.processing:
            return
        
        self.processing = True
        logger.info("Started real-time event processing")
        
        while self.processing:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                await self._process_event(event)
                self.connection_manager.events_processed += 1
                
            except asyncio.TimeoutError:
                # No events in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def stop_processing(self):
        """Stop the event processing loop"""
        self.processing = False
        logger.info("Stopped real-time event processing")
    
    async def queue_event(self, event: StreamEvent) -> bool:
        """Queue an event for processing"""
        try:
            self.event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.event_type.value}")
            return False
    
    async def _process_event(self, event: StreamEvent):
        """Process a single event"""
        handler = self.event_handlers.get(event.event_type)
        
        if handler:
            await handler(event)
        else:
            logger.warning(f"No handler for event type: {event.event_type.value}")
    
    async def _handle_health_data_update(self, event: StreamEvent):
        """Handle health data update events"""
        # Broadcast to all users interested in this location
        if event.location_id:
            sent_count = await self.connection_manager.broadcast_to_location(event.location_id, event)
            logger.info(f"Health data update sent to {sent_count} connections for location {event.location_id}")
    
    async def _handle_alert_notification(self, event: StreamEvent):
        """Handle alert notification events"""
        # Send to specific user if specified
        if event.user_id:
            sent_count = await self.connection_manager.send_to_user(event.user_id, event)
            logger.info(f"Alert sent to {sent_count} connections for user {event.user_id}")
        
        # Also broadcast to location if it's a location-wide alert
        elif event.location_id:
            sent_count = await self.connection_manager.broadcast_to_location(event.location_id, event)
            logger.info(f"Location alert sent to {sent_count} connections for location {event.location_id}")
    
    async def _handle_ml_prediction_update(self, event: StreamEvent):
        """Handle ML prediction update events"""
        # Broadcast to interested users
        if event.location_id:
            sent_count = await self.connection_manager.broadcast_to_location(event.location_id, event)
            logger.info(f"ML prediction update sent to {sent_count} connections")
    
    async def _handle_system_status(self, event: StreamEvent):
        """Handle system status events"""
        # Broadcast to all admin users
        sent_count = await self.connection_manager.broadcast_to_all(event)
        logger.info(f"System status update sent to {sent_count} connections")
    
    async def _handle_user_activity(self, event: StreamEvent):
        """Handle user activity events"""
        # Send only to the specific user
        if event.user_id:
            sent_count = await self.connection_manager.send_to_user(event.user_id, event)
            logger.info(f"User activity update sent to {sent_count} connections")

class HealthDataStreamer:
    """Streams real-time health data updates"""
    
    def __init__(self, event_processor: EventProcessor):
        self.event_processor = event_processor
        self.streaming = False
        self.last_data_check = {}  # location_id -> last_timestamp
    
    async def start_streaming(self, interval_seconds: int = 30):
        """Start streaming health data updates"""
        if self.streaming:
            return
        
        self.streaming = True
        logger.info(f"Started health data streaming with {interval_seconds}s interval")
        
        while self.streaming:
            try:
                await self._check_and_stream_updates()
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in health data streaming: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def stop_streaming(self):
        """Stop streaming health data updates"""
        self.streaming = False
        logger.info("Stopped health data streaming")
    
    async def _check_and_stream_updates(self):
        """Check for new health data and stream updates"""
        db = next(get_db())
        try:
            # Get all locations
            locations = db_service.health_data.get_locations(db)
            
            for location in locations:
                location_id = str(location.id)
                
                # Get latest data for this location
                latest_data = db_service.health_data.get_current_metrics(db, location_id)
                
                if not latest_data:
                    continue
                
                # Check if this is new data
                last_timestamp = self.last_data_check.get(location_id)
                current_timestamp = datetime.utcnow()
                
                if not last_timestamp or (current_timestamp - last_timestamp).total_seconds() > 30:
                    # Stream the update
                    await self.event_processor.queue_event(StreamEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=EventType.HEALTH_DATA_UPDATE,
                        timestamp=current_timestamp,
                        user_id=None,
                        location_id=location_id,
                        data={
                            'location': location.name,
                            'metrics': latest_data,
                            'updated_at': current_timestamp.isoformat()
                        },
                        priority='normal'
                    ))
                    
                    self.last_data_check[location_id] = current_timestamp
                    
        finally:
            db.close()

class AlertStreamer:
    """Streams real-time alert notifications"""
    
    def __init__(self, event_processor: EventProcessor):
        self.event_processor = event_processor
        self.streaming = False
        self.last_alert_check = datetime.utcnow() - timedelta(hours=1)
    
    async def start_streaming(self, interval_seconds: int = 10):
        """Start streaming alert notifications"""
        if self.streaming:
            return
        
        self.streaming = True
        logger.info(f"Started alert streaming with {interval_seconds}s interval")
        
        while self.streaming:
            try:
                await self._check_and_stream_alerts()
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in alert streaming: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def stop_streaming(self):
        """Stop streaming alert notifications"""
        self.streaming = False
        logger.info("Stopped alert streaming")
    
    async def _check_and_stream_alerts(self):
        """Check for new alerts and stream notifications"""
        db = next(get_db())
        try:
            # Get recent alerts
            recent_alerts = db_service.alerts.get_recent_alerts(
                db, since=self.last_alert_check, limit=50
            )
            
            for alert in recent_alerts:
                # Stream the alert
                await self.event_processor.queue_event(StreamEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.ALERT_NOTIFICATION,
                    timestamp=alert['timestamp'],
                    user_id=alert.get('user_id'),
                    location_id=alert.get('location_id'),
                    data={
                        'alert_id': alert.get('id'),
                        'title': alert.get('title', 'Health Alert'),
                        'message': alert['message'],
                        'severity': alert['severity'],
                        'location': alert.get('location'),
                        'metric_type': alert.get('metric_type')
                    },
                    priority='high' if alert['severity'] == 'HIGH' else 'normal'
                ))
            
            if recent_alerts:
                self.last_alert_check = max(alert['timestamp'] for alert in recent_alerts)
                
        finally:
            db.close()

class WebSocketServer:
    """Main WebSocket server for real-time streaming"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5001):
        self.host = host
        self.port = port
        self.connection_manager = ConnectionManager()
        self.event_processor = EventProcessor(self.connection_manager)
        self.health_streamer = HealthDataStreamer(self.event_processor)
        self.alert_streamer = AlertStreamer(self.event_processor)
        self.server = None
        self.running = False
    
    async def start_server(self):
        """Start the WebSocket server"""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available. Cannot start server.")
            return False
        
        try:
            # Start event processing
            asyncio.create_task(self.event_processor.start_processing())
            
            # Start data streamers
            asyncio.create_task(self.health_streamer.start_streaming(30))  # 30-second intervals
            asyncio.create_task(self.alert_streamer.start_streaming(10))   # 10-second intervals
            
            # Start WebSocket server
            self.server = await websockets.serve(
                self.handle_client, 
                self.host, 
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.running = True
            logger.info(f"WebSocket server started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        self.running = False
        
        # Stop streamers
        await self.health_streamer.stop_streaming()
        await self.alert_streamer.stop_streaming()
        await self.event_processor.stop_processing()
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle individual client connections"""
        connection_id = None
        
        try:
            # Get initial message with user authentication
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=10)
            auth_data = json.loads(auth_message)
            
            user_id = auth_data.get('user_id')
            if not user_id:
                await websocket.send(json.dumps({'error': 'Authentication required'}))
                return
            
            # Register connection
            connection_id = await self.connection_manager.connect(websocket, user_id, {
                'path': path,
                'subscribed_locations': auth_data.get('locations', []),
                'subscribed_events': auth_data.get('events', [])
            })
            
            # Subscribe to requested locations and events
            for location_id in auth_data.get('locations', []):
                await self.connection_manager.subscribe_to_location(connection_id, location_id)
            
            event_types = [EventType(event) for event in auth_data.get('events', []) if event in [e.value for e in EventType]]
            await self.connection_manager.subscribe_to_events(connection_id, event_types)
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(connection_id, data)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from connection {connection_id}")
                except Exception as e:
                    logger.error(f"Error handling message from {connection_id}: {e}")
        
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
        
        finally:
            if connection_id:
                await self.connection_manager.disconnect(connection_id)
    
    async def _handle_client_message(self, connection_id: str, data: Dict[str, Any]):
        """Handle messages from clients"""
        message_type = data.get('type')
        
        if message_type == 'ping':
            # Update last ping time
            if connection_id in self.connection_manager.connection_metadata:
                self.connection_manager.connection_metadata[connection_id]['last_ping'] = datetime.utcnow()
            
            # Send pong response
            await self.connection_manager.send_to_connection(connection_id, StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.CONNECTION_STATUS,
                timestamp=datetime.utcnow(),
                user_id=None,
                location_id=None,
                data={'type': 'pong'}
            ))
        
        elif message_type == 'subscribe_location':
            location_id = data.get('location_id')
            if location_id:
                await self.connection_manager.subscribe_to_location(connection_id, location_id)
        
        elif message_type == 'subscribe_events':
            events = data.get('events', [])
            event_types = [EventType(event) for event in events if event in [e.value for e in EventType]]
            await self.connection_manager.subscribe_to_events(connection_id, event_types)
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status and statistics"""
        return {
            'running': self.running,
            'host': self.host,
            'port': self.port,
            'websockets_available': WEBSOCKETS_AVAILABLE,
            **self.connection_manager.get_connection_stats()
        }

# Global WebSocket server instance
websocket_server = WebSocketServer()

# Convenience functions for triggering events from other parts of the application
async def broadcast_health_update(location_id: str, metrics: Dict[str, Any]):
    """Broadcast health data update to connected clients"""
    if websocket_server.running:
        await websocket_server.event_processor.queue_event(StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.HEALTH_DATA_UPDATE,
            timestamp=datetime.utcnow(),
            user_id=None,
            location_id=location_id,
            data={
                'location_id': location_id,
                'metrics': metrics,
                'updated_at': datetime.utcnow().isoformat()
            }
        ))

async def send_alert_notification(user_id: str, alert_data: Dict[str, Any]):
    """Send alert notification to specific user"""
    if websocket_server.running:
        await websocket_server.event_processor.queue_event(StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ALERT_NOTIFICATION,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            location_id=alert_data.get('location_id'),
            data=alert_data,
            priority='high'
        ))

async def broadcast_ml_prediction(location_id: str, predictions: Dict[str, Any]):
    """Broadcast ML prediction update"""
    if websocket_server.running:
        await websocket_server.event_processor.queue_event(StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ML_PREDICTION_UPDATE,
            timestamp=datetime.utcnow(),
            user_id=None,
            location_id=location_id,
            data={
                'location_id': location_id,
                'predictions': predictions,
                'generated_at': datetime.utcnow().isoformat()
            }
        ))

def start_websocket_server_background():
    """Start WebSocket server in background thread"""
    if not WEBSOCKETS_AVAILABLE:
        logger.warning("WebSocket server not started - websockets package not available")
        return
    
    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_server.start_server())
        loop.run_forever()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    logger.info("WebSocket server started in background thread")
#!/usr/bin/env python3
"""
Dashboard Health Monitoring Module for CryptoVisionAI

This module provides health check endpoints and system monitoring 
for the CryptoVisionAI dashboard.
"""

import os
import sys
import time
import json
import socket
import logging
import platform
import threading
import traceback
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import collections
import smtplib
from email.message import EmailMessage

# System monitoring
import psutil
from dataclasses import dataclass, field, asdict

# Configure logging
logger = logging.getLogger("dashboard_monitor")

@dataclass
class SystemMetrics:
    """Data class for system metrics"""
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    
    # Memory metrics
    memory_total: int = 0
    memory_available: int = 0
    memory_used: int = 0
    memory_percent: float = 0.0
    
    # Disk metrics
    disk_total: int = 0
    disk_used: int = 0
    disk_free: int = 0
    disk_percent: float = 0.0
    
    # Network metrics
    network_sent_bytes: int = 0
    network_recv_bytes: int = 0
    
    # Application metrics
    api_latency_ms: float = 0.0
    prediction_accuracy: float = 0.0
    active_connections: int = 0
    requests_per_minute: float = 0.0
    
    # System info
    hostname: str = ""
    platform_system: str = ""
    platform_release: str = ""
    python_version: str = ""
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper formatting"""
        data = asdict(self)
        # Format memory and disk as MB
        for key in ["memory_total", "memory_available", "memory_used"]:
            data[key] = f"{data[key] / (1024 * 1024):.2f} MB"
        
        for key in ["disk_total", "disk_used", "disk_free"]:
            data[key] = f"{data[key] / (1024 * 1024 * 1024):.2f} GB"
        
        # Format network as KB
        for key in ["network_sent_bytes", "network_recv_bytes"]:
            data[key] = f"{data[key] / 1024:.2f} KB"
            
        # Format percentages
        for key in ["cpu_percent", "memory_percent", "disk_percent"]:
            data[key] = f"{data[key]:.1f}%"
            
        # Format application metrics
        data["api_latency_ms"] = f"{data['api_latency_ms']:.2f} ms"
        data["prediction_accuracy"] = f"{data['prediction_accuracy']:.2f}%"
        data["requests_per_minute"] = f"{data['requests_per_minute']:.2f} rpm"
            
        return data

@dataclass
class HealthStatus:
    """Data class for health status"""
    status: str = "healthy"
    uptime: str = "0:00:00"
    services: Dict[str, str] = field(default_factory=dict)
    components: Dict[str, str] = field(default_factory=dict)
    metrics: SystemMetrics = field(default_factory=SystemMetrics)
    last_errors: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status,
            "uptime": self.uptime,
            "services": self.services,
            "components": self.components,
            "metrics": self.metrics.to_dict(),
            "last_errors": self.last_errors[:5],  # Only show last 5 errors
            "alerts": self.alerts[:5]  # Only show last 5 alerts
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class DashboardMonitor:
    """
    Health monitoring for the Dashboard application
    """
    
    def __init__(self, update_interval: int = 30, history_size: int = 60):
        """
        Initialize the dashboard monitor
        
        Args:
            update_interval: How often to update metrics (seconds)
            history_size: Number of historical data points to keep (default 60 = 30 minutes at 30s interval)
        """
        self.start_time = datetime.now()
        self.metrics = SystemMetrics()
        self.health = HealthStatus()
        self.errors = []
        self.alerts = []
        self.lock = threading.RLock()
        self.update_interval = update_interval
        self.update_thread = None
        self.running = False
        self.history_size = history_size
        
        # Historical metrics storage (deque for efficient fixed-size collection)
        self.metrics_history = collections.deque(maxlen=history_size)
        self.alert_callbacks = []
        
        # Alert thresholds
        self.thresholds = {
            "cpu_percent": 80.0,        # 80% CPU usage
            "memory_percent": 85.0,     # 85% memory usage
            "disk_percent": 90.0,       # 90% disk usage
            "api_latency_ms": 500.0,    # 500ms API latency
        }
        
        # Register components we'll monitor
        self.health.components = {
            "dashboard": "unknown",
            "data_processor": "unknown",
            "model_service": "unknown"
        }
        
        # Register services
        self.health.services = {
            "database": "unknown",
            "api": "unknown"
        }
        
        # Initialize with first update
        self._update_metrics()
    
    def start(self) -> None:
        """
        Start the monitor with background metrics collection
        """
        if self.running:
            return
            
        self.running = True
        self.update_thread = threading.Thread(
            target=self._metrics_collector_thread, 
            daemon=True
        )
        self.update_thread.start()
        logger.info("Dashboard monitor started")
    
    def stop(self) -> None:
        """
        Stop the monitor
        """
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        logger.info("Dashboard monitor stopped")
    
    def register_alert_callback(self, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Register a callback function to be called when alerts are triggered
        
        Args:
            callback: Function that takes (metric_name, current_value, threshold_value)
        """
        self.alert_callbacks.append(callback)
    
    def send_email_alert(self, subject: str, body: str) -> bool:
        """
        Send an email alert
        
        Args:
            subject: Email subject
            body: Email body
            
        Returns:
            True if email was sent successfully
        """
        try:
            # Get email settings from environment variables or use defaults
            smtp_server = os.environ.get("ALERT_SMTP_SERVER", "")
            smtp_port = int(os.environ.get("ALERT_SMTP_PORT", "587"))
            sender = os.environ.get("ALERT_EMAIL_SENDER", "")
            recipient = os.environ.get("ALERT_EMAIL_RECIPIENT", "")
            password = os.environ.get("ALERT_EMAIL_PASSWORD", "")
            
            # If email settings aren't configured, just log the alert
            if not all([smtp_server, smtp_port, sender, recipient, password]):
                logger.warning(f"Email alert not sent (not configured): {subject}")
                return False
                
            # Create and send the email
            msg = EmailMessage()
            msg.set_content(body)
            msg["Subject"] = f"CryptoVisionAI Alert: {subject}"
            msg["From"] = sender
            msg["To"] = recipient
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False

    def _metrics_collector_thread(self) -> None:
        """
        Background thread to collect metrics periodically
        """
        while self.running:
            try:
                self._update_metrics()
                self._check_thresholds()
                
                # Store metrics history
                with self.lock:
                    # Store a copy of the current metrics
                    self.metrics_history.append(asdict(self.metrics))
                    
            except Exception as e:
                self._record_error(f"Error updating metrics: {str(e)}")
            
            # Sleep until next update
            time.sleep(self.update_interval)
    
    def _update_metrics(self) -> None:
        """
        Update all system metrics
        """
        # Use threading lock to prevent race conditions
        with self.lock:
            # Update system info once
            if not self.metrics.hostname:
                self.metrics.hostname = socket.gethostname()
                self.metrics.platform_system = platform.system()
                self.metrics.platform_release = platform.release()
                self.metrics.python_version = sys.version.split()[0]
            
            # Update CPU metrics
            self.metrics.cpu_count = psutil.cpu_count(logical=True)
            self.metrics.cpu_percent = psutil.cpu_percent(interval=0.5)
            
            # Update memory metrics
            mem = psutil.virtual_memory()
            self.metrics.memory_total = mem.total
            self.metrics.memory_available = mem.available
            self.metrics.memory_used = mem.used
            self.metrics.memory_percent = mem.percent
            
            # Update disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.disk_total = disk.total
            self.metrics.disk_used = disk.used
            self.metrics.disk_free = disk.free
            self.metrics.disk_percent = disk.percent
            
            # Update network metrics
            network = psutil.net_io_counters()
            self.metrics.network_sent_bytes = network.bytes_sent
            self.metrics.network_recv_bytes = network.bytes_recv
            
            # Update timestamp
            self.metrics.timestamp = datetime.now().isoformat()
            
            # Calculate uptime
            uptime = datetime.now() - self.start_time
            self.health.uptime = str(uptime).split('.')[0]  # Remove microseconds
            
            # Update health in the status object
            self.health.metrics = self.metrics
    
    def _check_thresholds(self) -> None:
        """
        Check if any metrics have exceeded their threshold values
        """
        alerts = []
        
        with self.lock:
            # Check CPU usage
            if self.metrics.cpu_percent > self.thresholds["cpu_percent"]:
                alert_msg = f"High CPU usage: {self.metrics.cpu_percent:.1f}% (threshold: {self.thresholds['cpu_percent']}%)"
                alerts.append(alert_msg)
                self._trigger_alert("cpu_percent", self.metrics.cpu_percent, self.thresholds["cpu_percent"])
                
            # Check memory usage
            if self.metrics.memory_percent > self.thresholds["memory_percent"]:
                alert_msg = f"High memory usage: {self.metrics.memory_percent:.1f}% (threshold: {self.thresholds['memory_percent']}%)"
                alerts.append(alert_msg)
                self._trigger_alert("memory_percent", self.metrics.memory_percent, self.thresholds["memory_percent"])
                
            # Check disk usage
            if self.metrics.disk_percent > self.thresholds["disk_percent"]:
                alert_msg = f"High disk usage: {self.metrics.disk_percent:.1f}% (threshold: {self.thresholds['disk_percent']}%)"
                alerts.append(alert_msg)
                self._trigger_alert("disk_percent", self.metrics.disk_percent, self.thresholds["disk_percent"])
                
            # Check API latency
            if self.metrics.api_latency_ms > self.thresholds["api_latency_ms"]:
                alert_msg = f"High API latency: {self.metrics.api_latency_ms:.1f}ms (threshold: {self.thresholds['api_latency_ms']}ms)"
                alerts.append(alert_msg)
                self._trigger_alert("api_latency_ms", self.metrics.api_latency_ms, self.thresholds["api_latency_ms"])
        
        # Record alerts with timestamp
        if alerts:
            timestamp = datetime.now().isoformat()
            for alert in alerts:
                self.alerts.append(f"[{timestamp}] {alert}")
                logger.warning(alert)
            
            # Update health status with alerts
            with self.lock:
                self.health.alerts = self.alerts[-10:]  # Keep the 10 most recent alerts
    
    def _trigger_alert(self, metric_name: str, current_value: Any, threshold_value: Any) -> None:
        """
        Trigger alert callbacks for a metric exceeding threshold
        
        Args:
            metric_name: Name of the metric
            current_value: Current value of the metric
            threshold_value: Threshold value that was exceeded
        """
        # Call all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(metric_name, current_value, threshold_value)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def _record_error(self, error_msg: str) -> None:
        """
        Record an error in the error log
        
        Args:
            error_msg: Error message to record
        """
        timestamp = datetime.now().isoformat()
        self.errors.append(f"[{timestamp}] {error_msg}")
        logger.error(error_msg)
        
        with self.lock:
            # Update the health status with latest errors
            self.health.last_errors = self.errors[-10:]  # Keep last 10 errors
            
            # Only set status to unhealthy if we have recent errors (last 5 minutes)
            recent_errors = [e for e in self.errors 
                            if datetime.now() - datetime.fromisoformat(e.split(']')[0][1:]) < timedelta(minutes=5)]
            
            if recent_errors:
                self.health.status = "unhealthy"
            else:
                self.health.status = "healthy"
    
    def update_component_status(self, component: str, status: str) -> None:
        """
        Update the status of a component
        
        Args:
            component: Component name
            status: Status value
        """
        with self.lock:
            if component in self.health.components:
                self.health.components[component] = status
    
    def update_service_status(self, service: str, status: str) -> None:
        """
        Update the status of a service
        
        Args:
            service: Service name
            status: Status value
        """
        with self.lock:
            if service in self.health.services:
                self.health.services[service] = status
    
    def update_application_metrics(self, 
                                  api_latency_ms: Optional[float] = None,
                                  prediction_accuracy: Optional[float] = None,
                                  active_connections: Optional[int] = None,
                                  requests_per_minute: Optional[float] = None) -> None:
        """
        Update application-specific metrics
        
        Args:
            api_latency_ms: API response time in milliseconds
            prediction_accuracy: Model prediction accuracy as percentage
            active_connections: Number of active dashboard connections
            requests_per_minute: API requests per minute
        """
        with self.lock:
            if api_latency_ms is not None:
                self.metrics.api_latency_ms = api_latency_ms
            if prediction_accuracy is not None:
                self.metrics.prediction_accuracy = prediction_accuracy
            if active_connections is not None:
                self.metrics.active_connections = active_connections
            if requests_per_minute is not None:
                self.metrics.requests_per_minute = requests_per_minute
    
    def set_threshold(self, metric_name: str, threshold_value: float) -> None:
        """
        Set or update a metric threshold
        
        Args:
            metric_name: Name of the metric
            threshold_value: New threshold value
        """
        with self.lock:
            self.thresholds[metric_name] = threshold_value
            logger.info(f"Set {metric_name} threshold to {threshold_value}")
    
    def get_metrics_history(self, 
                           metrics: Optional[List[str]] = None, 
                           minutes: Optional[int] = None) -> Dict[str, List[Any]]:
        """
        Get historical metrics data
        
        Args:
            metrics: List of metric names to include (None for all)
            minutes: Number of minutes of history to include (None for all available)
            
        Returns:
            Dictionary with metric names as keys and lists of values as values
        """
        with self.lock:
            # Start with a copy of the metrics history
            history = list(self.metrics_history)
            
            # Filter by time if specified
            if minutes is not None:
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                history = [
                    entry for entry in history 
                    if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
                ]
            
            # Extract the requested metrics
            result = {
                "timestamps": [entry["timestamp"] for entry in history]
            }
            
            if metrics is None:
                # Include all numeric metrics
                for key in history[0].keys() if history else []:
                    if key != "timestamp" and key not in ["hostname", "platform_system", "platform_release", "python_version"]:
                        result[key] = [entry[key] for entry in history]
            else:
                # Include only specified metrics
                for key in metrics:
                    if history and key in history[0]:
                        result[key] = [entry[key] for entry in history]
            
            return result
    
    def get_health(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Get the current health status
        
        Args:
            detailed: Whether to include detailed metrics
            
        Returns:
            Dict containing health information
        """
        with self.lock:
            # Create a basic health response
            health_data = {
                "status": self.health.status,
                "uptime": self.health.uptime,
            }
            
            # Include more details if requested
            if detailed:
                health_data.update({
                    "services": self.health.services,
                    "components": self.health.components,
                    "metrics": self.metrics.to_dict(),
                    "last_errors": self.health.last_errors,
                    "alerts": self.health.alerts
                })
                
            return health_data
    
    def export_metrics_csv(self, file_path: str, include_all: bool = False) -> bool:
        """
        Export metrics history to a CSV file
        
        Args:
            file_path: Path to write the CSV file
            include_all: Whether to include all metrics or just core metrics
            
        Returns:
            True if successful
        """
        try:
            import csv
            
            with self.lock:
                history = list(self.metrics_history)
                
                if not history:
                    logger.warning("No metrics history to export")
                    return False
                
                # Determine which fields to export
                core_fields = ["timestamp", "cpu_percent", "memory_percent", 
                              "disk_percent", "api_latency_ms"]
                
                if include_all:
                    # Include all numeric metrics
                    fieldnames = [k for k in history[0].keys() if k not in 
                                 ["hostname", "platform_system", "platform_release", "python_version"]]
                else:
                    fieldnames = core_fields
                
                # Write to CSV
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for entry in history:
                        row = {field: entry[field] for field in fieldnames}
                        writer.writerow(row)
                
                logger.info(f"Exported metrics history to {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return False


# Global instance for easy import
monitor = None

def get_monitor() -> DashboardMonitor:
    """
    Get or create the dashboard monitor instance
    
    Returns:
        DashboardMonitor instance
    """
    global monitor
    if monitor is None:
        monitor = DashboardMonitor()
        monitor.start()
    return monitor

def email_alert_callback(metric_name: str, current_value: Any, threshold_value: Any) -> None:
    """
    Callback function for email alerts
    
    Args:
        metric_name: Name of the metric
        current_value: Current value of the metric
        threshold_value: Threshold value that was exceeded
    """
    monitor = get_monitor()
    subject = f"Alert: {metric_name} threshold exceeded"
    body = f"""
CryptoVisionAI System Alert
---------------------------
Metric: {metric_name}
Current Value: {current_value}
Threshold: {threshold_value}
Time: {datetime.now().isoformat()}
Hostname: {monitor.metrics.hostname}

Please check the system status at the earliest opportunity.
    """
    monitor.send_email_alert(subject, body)

def setup_health_endpoints(app) -> DashboardMonitor:
    """
    Setup health check endpoints for a Dash application
    
    Args:
        app: Dash application instance
        
    Returns:
        DashboardMonitor instance
    """
    monitor = get_monitor()
    
    # Register the email alert callback
    monitor.register_alert_callback(email_alert_callback)
    
    @app.server.route('/health')
    def health_endpoint():
        """Basic health endpoint"""
        from flask import jsonify
        return jsonify(monitor.get_health(detailed=False))
        
    @app.server.route('/health/detailed')
    def detailed_health_endpoint():
        """Detailed health endpoint with all metrics"""
        from flask import jsonify
        return jsonify(monitor.get_health(detailed=True))
    
    @app.server.route('/health/history')
    def metrics_history_endpoint():
        """Get historical metrics data"""
        from flask import jsonify, request
        
        # Parse query parameters
        metrics = request.args.get('metrics', '').split(',') if request.args.get('metrics') else None
        minutes = int(request.args.get('minutes')) if request.args.get('minutes') else None
        
        return jsonify(monitor.get_metrics_history(metrics=metrics, minutes=minutes))
    
    logger.info("Health check endpoints registered at /health, /health/detailed, and /health/history")
    return monitor

def get_health_status() -> Dict[str, Any]:
    """
    Get the current health status of the system
    
    Returns:
        Dictionary with health status information
    """
    # Get the monitor singleton
    monitor = get_monitor()
    
    # Get the health data with full details
    health_data = monitor.get_health(detailed=True)
    
    # Format data in the way the dashboard expects it
    system_info = {
        "platform": monitor.metrics.platform_system,
        "platform_release": monitor.metrics.platform_release,
        "python_version": monitor.metrics.python_version,
        "architecture": platform.architecture()[0],
        "processor": platform.processor()
    }
    
    # Memory information
    memory_usage = {
        "system_memory_total_gb": monitor.metrics.memory_total / (1024 * 1024 * 1024),
        "system_memory_available_gb": monitor.metrics.memory_available / (1024 * 1024 * 1024),
        "system_memory_used_percent": monitor.metrics.memory_percent,
        "process_memory_rss_mb": psutil.Process().memory_info().rss / (1024 * 1024)
    }
    
    # CPU information
    cpu_usage = {
        "system_cpu_percent": monitor.metrics.cpu_percent,
        "process_cpu_percent": psutil.Process().cpu_percent(),
        "cpu_count": monitor.metrics.cpu_count
    }
    
    # Disk information
    disk_space = {
        "disk_total_gb": monitor.metrics.disk_total / (1024 * 1024 * 1024),
        "disk_free_gb": monitor.metrics.disk_free / (1024 * 1024 * 1024),
        "disk_used_gb": monitor.metrics.disk_used / (1024 * 1024 * 1024),
        "disk_used_percent": monitor.metrics.disk_percent
    }
    
    # Uptime information
    uptime_info = {
        "start_time": monitor.start_time.isoformat(),
        "uptime": monitor.health.uptime
    }
    
    # Create warnings list from alerts
    warnings = []
    for alert in monitor.health.alerts:
        # Extract just the message part without timestamp
        if ']' in alert:
            message = alert.split(']', 1)[1].strip()
            warnings.append(message)
        else:
            warnings.append(alert)
    
    # Complete health status
    status_data = {
        "status": monitor.health.status,
        "system_info": system_info,
        "memory_usage": memory_usage,
        "cpu_usage": cpu_usage,
        "disk_space": disk_space,
        "uptime": uptime_info,
        "warnings": warnings,
        "timestamp": datetime.now().isoformat()
    }
    
    return status_data

def get_simple_health() -> Dict[str, Any]:
    """
    Get a simplified health status
    
    Returns:
        Dictionary with basic health status
    """
    monitor = get_monitor()
    
    return {
        "status": monitor.health.status,
        "uptime": monitor.health.uptime,
        "timestamp": datetime.now().isoformat()
    }
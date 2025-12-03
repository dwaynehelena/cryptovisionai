"""
Monitoring module for CryptoVisionAI Dashboard

This module provides health check endpoints and monitoring capabilities for the dashboard.
"""

import os
import sys
import time
import json
import logging
import platform
import psutil
import datetime
from typing import Dict, Any

# Configure logging
logger = logging.getLogger("monitoring")

class DashboardMonitor:
    """Monitoring class for CryptoVisionAI Dashboard"""
    
    def __init__(self):
        """Initialize the dashboard monitor"""
        self.start_time = time.time()
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.health_checks = {
            "system_info": self.check_system_info,
            "memory_usage": self.check_memory_usage,
            "cpu_usage": self.check_cpu_usage,
            "disk_space": self.check_disk_space,
            "uptime": self.check_uptime
        }
        logger.info("Dashboard monitor initialized")
    
    def check_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage of the dashboard process"""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_rss_mb": memory_info.rss / 1024 / 1024,
            "process_memory_vms_mb": memory_info.vms / 1024 / 1024,
            "system_memory_total_gb": system_memory.total / 1024 / 1024 / 1024,
            "system_memory_available_gb": system_memory.available / 1024 / 1024 / 1024,
            "system_memory_used_percent": system_memory.percent
        }
    
    def check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage of the dashboard process"""
        return {
            "process_cpu_percent": self.process.cpu_percent(interval=0.1),
            "system_cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(logical=True)
        }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability"""
        disk_usage = psutil.disk_usage('/')
        return {
            "disk_total_gb": disk_usage.total / 1024 / 1024 / 1024,
            "disk_used_gb": disk_usage.used / 1024 / 1024 / 1024,
            "disk_free_gb": disk_usage.free / 1024 / 1024 / 1024,
            "disk_used_percent": disk_usage.percent
        }
    
    def check_uptime(self) -> Dict[str, Any]:
        """Check uptime of the dashboard"""
        uptime_seconds = time.time() - self.start_time
        uptime_str = str(datetime.timedelta(seconds=int(uptime_seconds)))
        return {
            "start_time": datetime.datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
            "uptime_seconds": uptime_seconds,
            "uptime": uptime_str
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Run all health checks
        try:
            for check_name, check_func in self.health_checks.items():
                health_status[check_name] = check_func()
            
            # Add simple warning indicators based on resource usage
            warnings = []
            
            if health_status["memory_usage"]["system_memory_used_percent"] > 90:
                warnings.append("High system memory usage")
                health_status["status"] = "warning"
            
            if health_status["cpu_usage"]["system_cpu_percent"] > 90:
                warnings.append("High CPU usage")
                health_status["status"] = "warning"
            
            if health_status["disk_space"]["disk_used_percent"] > 90:
                warnings.append("Low disk space")
                health_status["status"] = "warning"
            
            health_status["warnings"] = warnings
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            health_status["status"] = "error"
            health_status["error"] = str(e)
        
        return health_status
    
    def get_simple_health(self) -> Dict[str, Any]:
        """Get simplified health status for quick checks"""
        try:
            memory_usage = self.process.memory_info().rss / 1024 / 1024
            cpu_usage = self.process.cpu_percent(interval=0.1)
            uptime_seconds = time.time() - self.start_time
            
            return {
                "status": "healthy",
                "memory_mb": round(memory_usage, 2),
                "cpu_percent": round(cpu_usage, 2),
                "uptime_seconds": round(uptime_seconds, 2),
                "uptime": str(datetime.timedelta(seconds=int(uptime_seconds))),
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error during simple health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

# Global instance of the monitor
dashboard_monitor = DashboardMonitor()

def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status"""
    return dashboard_monitor.get_health_status()

def get_simple_health() -> Dict[str, Any]:
    """Get simplified health status"""
    return dashboard_monitor.get_simple_health()
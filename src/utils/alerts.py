#!/usr/bin/env python3
"""
Alerts Module - Comprehensive system for monitoring and alerting in the trading system
Provides email, SMS, and logging alerts for various risk levels
"""

import logging
import smtplib
import os
import json
import time
import threading
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union, Callable

# Configure module logger
logger = logging.getLogger("alerts")

class AlertLevel:
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class AlertType:
    """Types of alerts"""
    SYSTEM = "SYSTEM"           # System issues (connectivity, errors)
    PERFORMANCE = "PERFORMANCE"  # Performance metrics
    RISK = "RISK"               # Risk management alerts
    TRADING = "TRADING"         # Trading signals and executions
    SECURITY = "SECURITY"       # Security-related alerts
    MARKET = "MARKET"           # Market condition alerts
    POSITION = "POSITION"       # Position-related alerts

class AlertManager:
    """
    Central alert management system
    Handles different types of alerts and notifications
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager
        
        Args:
            config (Dict[str, Any]): Alert configuration
        """
        self.config = config
        self.enabled = config.get("alerts_enabled", True)
        self.test_mode = config.get("test_mode", True)
        self.alert_history = []
        self.max_history = config.get("max_history", 1000)
        
        # Alert thresholds
        self.thresholds = config.get("thresholds", {
            "max_drawdown_warning": 10.0,    # Portfolio drawdown % for warning
            "max_drawdown_critical": 15.0,   # Portfolio drawdown % for critical alert
            "max_drawdown_emergency": 20.0,  # Portfolio drawdown % for emergency alert
            "position_loss_warning": 5.0,    # Position loss % for warning
            "position_loss_critical": 10.0,  # Position loss % for critical alert
            "trade_size_warning": 10.0,      # Trade size % of portfolio for warning
            "trade_size_critical": 20.0,     # Trade size % of portfolio for critical alert
            "api_latency_warning": 500,      # API latency in ms for warning
            "api_latency_critical": 2000,    # API latency in ms for critical alert
            "error_rate_warning": 5,         # Errors per hour for warning
            "error_rate_critical": 20        # Errors per hour for critical alert
        })
        
        # Email configuration
        self.email_config = config.get("email", {})
        self.email_enabled = self.email_config.get("enabled", False)
        self.email_recipients = self.email_config.get("recipients", [])
        
        # SMS configuration
        self.sms_config = config.get("sms", {})
        self.sms_enabled = self.sms_config.get("enabled", False)
        self.sms_recipients = self.sms_config.get("recipients", [])
        
        # Webhook/API configuration
        self.webhook_config = config.get("webhook", {})
        self.webhook_enabled = self.webhook_config.get("enabled", False)
        self.webhook_url = self.webhook_config.get("url", "")
        
        # Load rate limiting settings
        self.rate_limits = config.get("rate_limits", {
            "email_interval_minutes": 15,
            "sms_interval_minutes": 30,
            "webhook_interval_minutes": 5
        })
        
        # Initialize rate limiters
        self._last_email_time = {}
        self._last_sms_time = {}
        self._last_webhook_time = {}
        
        # Error counter for monitoring system health
        self._error_count = 0
        self._error_timer = time.time()
        
        # Start background thread for timed alerts
        self._start_timed_alerts()
        
        logger.info("Alert manager initialized with email=%s, sms=%s", 
                   "enabled" if self.email_enabled else "disabled",
                   "enabled" if self.sms_enabled else "disabled")
    
    def alert(self, message: str, level: str = AlertLevel.INFO, 
             alert_type: str = AlertType.SYSTEM, data: Optional[Dict] = None) -> None:
        """
        Send an alert through configured channels
        
        Args:
            message (str): Alert message
            level (str): Alert level (INFO, WARNING, CRITICAL, EMERGENCY)
            alert_type (str): Type of alert
            data (Optional[Dict]): Additional data for the alert
        """
        if not self.enabled:
            return
        
        # Create alert object
        alert = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level,
            "type": alert_type,
            "data": data or {},
            "test_mode": self.test_mode
        }
        
        # Add to history
        self._add_to_history(alert)
        
        # Log the alert
        self._log_alert(alert)
        
        # For CRITICAL and EMERGENCY alerts, send notifications
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self._send_notifications(alert)
            
            # Count errors for health monitoring
            if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                self._error_count += 1
        
        # If in production mode, add additional notifications for WARNING+ level alerts
        if not self.test_mode and level in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self._send_notifications(alert)
    
    def position_alert(self, position_data: Dict[str, Any], 
                      level: str = AlertLevel.INFO) -> None:
        """
        Send an alert about a position
        
        Args:
            position_data (Dict[str, Any]): Position data
            level (str): Alert level
        """
        # Format a nice message for the position
        symbol = position_data.get("symbol", "Unknown")
        position_type = position_data.get("position_type", "Unknown")
        pnl = position_data.get("pnl", 0.0)
        pnl_percent = position_data.get("pnl_percent", 0.0)
        
        message = f"Position Alert: {symbol} {position_type} position "
        
        if position_data.get("status") == "closed":
            message += f"closed with P&L: {pnl:.2f} ({pnl_percent:.2f}%)"
        else:
            message += f"currently at P&L: {pnl:.2f} ({pnl_percent:.2f}%)"
        
        self.alert(message, level, AlertType.POSITION, position_data)
    
    def trading_alert(self, action: str, symbol: str, price: float, amount: float, 
                     level: str = AlertLevel.INFO, details: Optional[Dict] = None) -> None:
        """
        Send an alert about a trade
        
        Args:
            action (str): Trade action (buy, sell)
            symbol (str): Trading symbol
            price (float): Execution price
            amount (float): Trade amount
            level (str): Alert level
            details (Optional[Dict]): Additional details
        """
        message = f"Trade Alert: {action.upper()} {amount:.6f} {symbol} @ {price:.2f}"
        
        data = {
            "action": action,
            "symbol": symbol,
            "price": price,
            "amount": amount,
            **(details or {})
        }
        
        self.alert(message, level, AlertType.TRADING, data)
    
    def risk_alert(self, risk_type: str, value: float, threshold: float, 
                  level: str = AlertLevel.WARNING, details: Optional[Dict] = None) -> None:
        """
        Send a risk-related alert
        
        Args:
            risk_type (str): Type of risk (drawdown, exposure, etc.)
            value (float): Current risk value
            threshold (float): Risk threshold that was exceeded
            level (str): Alert level
            details (Optional[Dict]): Additional details
        """
        message = f"Risk Alert: {risk_type} at {value:.2f}% exceeds {threshold:.2f}% threshold"
        
        data = {
            "risk_type": risk_type, 
            "value": value,
            "threshold": threshold,
            **(details or {})
        }
        
        self.alert(message, level, AlertType.RISK, data)
    
    def system_alert(self, component: str, status: str, 
                    level: str = AlertLevel.INFO, details: Optional[Dict] = None) -> None:
        """
        Send a system status alert
        
        Args:
            component (str): System component
            status (str): Status message
            level (str): Alert level
            details (Optional[Dict]): Additional details
        """
        message = f"System Alert: {component} - {status}"
        
        data = {
            "component": component,
            "status": status,
            **(details or {})
        }
        
        self.alert(message, level, AlertType.SYSTEM, data)
    
    def market_alert(self, symbol: str, condition: str, 
                    level: str = AlertLevel.INFO, details: Optional[Dict] = None) -> None:
        """
        Send a market condition alert
        
        Args:
            symbol (str): Trading symbol
            condition (str): Market condition description
            level (str): Alert level
            details (Optional[Dict]): Additional details
        """
        message = f"Market Alert: {symbol} - {condition}"
        
        data = {
            "symbol": symbol,
            "condition": condition,
            **(details or {})
        }
        
        self.alert(message, level, AlertType.MARKET, data)
    
    def check_drawdown(self, drawdown_percent: float, portfolio_value: float) -> None:
        """
        Check drawdown against thresholds and send alert if exceeded
        
        Args:
            drawdown_percent (float): Current drawdown percentage
            portfolio_value (float): Current portfolio value
        """
        if drawdown_percent >= self.thresholds["max_drawdown_emergency"]:
            self.risk_alert("drawdown", drawdown_percent, 
                          self.thresholds["max_drawdown_emergency"],
                          AlertLevel.EMERGENCY, {"portfolio_value": portfolio_value})
        elif drawdown_percent >= self.thresholds["max_drawdown_critical"]:
            self.risk_alert("drawdown", drawdown_percent, 
                          self.thresholds["max_drawdown_critical"],
                          AlertLevel.CRITICAL, {"portfolio_value": portfolio_value})
        elif drawdown_percent >= self.thresholds["max_drawdown_warning"]:
            self.risk_alert("drawdown", drawdown_percent, 
                          self.thresholds["max_drawdown_warning"],
                          AlertLevel.WARNING, {"portfolio_value": portfolio_value})
    
    def check_position_loss(self, position_data: Dict[str, Any]) -> None:
        """
        Check position loss against thresholds and send alert if exceeded
        
        Args:
            position_data (Dict[str, Any]): Position data
        """
        pnl_percent = position_data.get("pnl_percent", 0.0)
        
        # Only check for negative P&L (losses)
        if pnl_percent >= 0:
            return
            
        # Convert to positive number for comparison
        loss_percent = abs(pnl_percent)
        
        if loss_percent >= self.thresholds["position_loss_critical"]:
            self.position_alert(position_data, AlertLevel.CRITICAL)
        elif loss_percent >= self.thresholds["position_loss_warning"]:
            self.position_alert(position_data, AlertLevel.WARNING)
    
    def check_trade_size(self, trade_value: float, portfolio_value: float, 
                        symbol: str, action: str) -> None:
        """
        Check if trade size exceeds thresholds
        
        Args:
            trade_value (float): Value of the trade
            portfolio_value (float): Current portfolio value
            symbol (str): Trading symbol
            action (str): Trade action
        """
        if portfolio_value <= 0:
            return
            
        trade_percent = (trade_value / portfolio_value) * 100
        
        if trade_percent >= self.thresholds["trade_size_critical"]:
            self.trading_alert(action, symbol, 0, 0, AlertLevel.CRITICAL,
                             {"trade_percent": trade_percent, 
                              "trade_value": trade_value,
                              "portfolio_value": portfolio_value})
        elif trade_percent >= self.thresholds["trade_size_warning"]:
            self.trading_alert(action, symbol, 0, 0, AlertLevel.WARNING,
                             {"trade_percent": trade_percent, 
                              "trade_value": trade_value,
                              "portfolio_value": portfolio_value})
    
    def check_api_latency(self, latency_ms: float, api_name: str) -> None:
        """
        Check if API latency exceeds thresholds
        
        Args:
            latency_ms (float): API latency in milliseconds
            api_name (str): Name of the API
        """
        if latency_ms >= self.thresholds["api_latency_critical"]:
            self.system_alert(api_name, f"High latency: {latency_ms}ms", 
                            AlertLevel.CRITICAL, {"latency": latency_ms})
        elif latency_ms >= self.thresholds["api_latency_warning"]:
            self.system_alert(api_name, f"Elevated latency: {latency_ms}ms", 
                            AlertLevel.WARNING, {"latency": latency_ms})
    
    def _add_to_history(self, alert: Dict[str, Any]) -> None:
        """
        Add alert to history, maintaining maximum size
        
        Args:
            alert (Dict[str, Any]): Alert object
        """
        self.alert_history.append(alert)
        
        # Trim history if it exceeds max size
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
    
    def _log_alert(self, alert: Dict[str, Any]) -> None:
        """
        Log the alert using the logging system
        
        Args:
            alert (Dict[str, Any]): Alert object
        """
        log_message = f"[{alert['level']}] {alert['type']}: {alert['message']}"
        
        if alert['level'] == AlertLevel.EMERGENCY:
            logger.critical(log_message)
        elif alert['level'] == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif alert['level'] == AlertLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _send_notifications(self, alert: Dict[str, Any]) -> None:
        """
        Send notifications through configured channels
        
        Args:
            alert (Dict[str, Any]): Alert object
        """
        # Send email notifications for critical/emergency alerts
        if self.email_enabled:
            self._send_email_alert(alert)
        
        # Send SMS notifications for emergency alerts
        if self.sms_enabled and alert['level'] in [AlertLevel.EMERGENCY]:
            self._send_sms_alert(alert)
        
        # Send webhook notifications
        if self.webhook_enabled:
            self._send_webhook_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """
        Send email alert
        
        Args:
            alert (Dict[str, Any]): Alert object
        """
        # Check rate limiting
        alert_key = f"{alert['type']}_{alert['level']}"
        current_time = time.time()
        
        # If this type of alert was sent recently, skip
        if alert_key in self._last_email_time:
            minutes_since_last = (current_time - self._last_email_time[alert_key]) / 60
            if minutes_since_last < self.rate_limits["email_interval_minutes"]:
                logger.info(f"Email alert rate limited: {alert_key}")
                return
        
        # Update last sent time
        self._last_email_time[alert_key] = current_time
        
        try:
            # Skip actual sending in test mode
            if self.test_mode:
                logger.info(f"Would send email in production mode: {alert['message']}")
                return
                
            if not self.email_recipients:
                logger.warning("No email recipients configured")
                return
                
            # Get SMTP settings
            smtp_server = self.email_config.get("smtp_server")
            smtp_port = self.email_config.get("smtp_port", 587)
            smtp_user = self.email_config.get("smtp_user")
            smtp_password = self.email_config.get("smtp_password")
            sender = self.email_config.get("sender", smtp_user)
            
            if not all([smtp_server, smtp_user, smtp_password]):
                logger.warning("Incomplete email configuration, can't send emails")
                return
            
            # Create email message
            subject = f"CryptoVisionAI Alert: {alert['level']} - {alert['type']}"
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ", ".join(self.email_recipients)
            msg['Subject'] = subject
            
            # Create email body
            body = f"""<html>
            <body>
                <h2>{alert['level']} Alert - {alert['type']}</h2>
                <p><strong>Message:</strong> {alert['message']}</p>
                <p><strong>Time:</strong> {alert['timestamp']}</p>
                <h3>Additional Data:</h3>
                <pre>{json.dumps(alert['data'], indent=2)}</pre>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
    
    def _send_sms_alert(self, alert: Dict[str, Any]) -> None:
        """
        Send SMS alert
        
        Args:
            alert (Dict[str, Any]): Alert object
        """
        # Check rate limiting
        alert_key = f"{alert['type']}_{alert['level']}"
        current_time = time.time()
        
        # If this type of alert was sent recently, skip
        if alert_key in self._last_sms_time:
            minutes_since_last = (current_time - self._last_sms_time[alert_key]) / 60
            if minutes_since_last < self.rate_limits["sms_interval_minutes"]:
                logger.info(f"SMS alert rate limited: {alert_key}")
                return
        
        # Update last sent time
        self._last_sms_time[alert_key] = current_time
        
        try:
            # Skip actual sending in test mode
            if self.test_mode:
                logger.info(f"Would send SMS in production mode: {alert['message']}")
                return
                
            if not self.sms_recipients:
                logger.warning("No SMS recipients configured")
                return
            
            # Get SMS settings (this would integrate with a service like Twilio)
            # For now, just log that we would send SMS
            logger.info(f"Would send SMS to {len(self.sms_recipients)} recipients: {alert['message']}")
            
            # Here you would integrate with an SMS service API
            # For example, with Twilio:
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)
            # for recipient in self.sms_recipients:
            #     message = client.messages.create(
            #         body=f"{alert['level']}: {alert['message']}",
            #         from_='+1234567890',
            #         to=recipient
            #     )
            
        except Exception as e:
            logger.error(f"Error sending SMS alert: {str(e)}")
    
    def _send_webhook_alert(self, alert: Dict[str, Any]) -> None:
        """
        Send webhook alert
        
        Args:
            alert (Dict[str, Any]): Alert object
        """
        # Check rate limiting
        alert_key = f"{alert['type']}_{alert['level']}"
        current_time = time.time()
        
        # If this type of alert was sent recently, skip
        if alert_key in self._last_webhook_time:
            minutes_since_last = (current_time - self._last_webhook_time[alert_key]) / 60
            if minutes_since_last < self.rate_limits["webhook_interval_minutes"]:
                logger.info(f"Webhook alert rate limited: {alert_key}")
                return
        
        # Update last sent time
        self._last_webhook_time[alert_key] = current_time
        
        try:
            # Skip actual sending in test mode
            if self.test_mode:
                logger.info(f"Would send webhook in production mode: {alert['message']}")
                return
                
            if not self.webhook_url:
                logger.warning("No webhook URL configured")
                return
            
            # Here you would send the webhook
            # For example:
            # import requests
            # response = requests.post(
            #     self.webhook_url,
            #     json=alert,
            #     headers={"Content-Type": "application/json"}
            # )
            # if response.status_code != 200:
            #     logger.warning(f"Webhook returned non-200 status: {response.status_code}")
            
            logger.info(f"Would send webhook to {self.webhook_url}")
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {str(e)}")
    
    def _start_timed_alerts(self) -> None:
        """Start background thread for timed alerts"""
        def check_system_health():
            while True:
                try:
                    # Check error rate
                    current_time = time.time()
                    elapsed_hours = (current_time - self._error_timer) / 3600
                    
                    if elapsed_hours >= 1:  # Check every hour
                        error_rate = self._error_count
                        
                        if error_rate >= self.thresholds["error_rate_critical"]:
                            self.system_alert("ErrorMonitor", f"High error rate: {error_rate}/hour", 
                                           AlertLevel.CRITICAL, {"error_rate": error_rate})
                        elif error_rate >= self.thresholds["error_rate_warning"]:
                            self.system_alert("ErrorMonitor", f"Elevated error rate: {error_rate}/hour", 
                                           AlertLevel.WARNING, {"error_rate": error_rate})
                        
                        # Reset counters
                        self._error_count = 0
                        self._error_timer = current_time
                    
                    # Sleep for a while
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in health check thread: {str(e)}")
                    time.sleep(300)  # Sleep for 5 minutes on error
        
        # Start thread
        health_thread = threading.Thread(target=check_system_health)
        health_thread.daemon = True
        health_thread.start()
    
    def get_alert_history(self, 
                         level: Optional[str] = None, 
                         alert_type: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get alert history, optionally filtered
        
        Args:
            level (Optional[str]): Filter by alert level
            alert_type (Optional[str]): Filter by alert type
            limit (int): Maximum number of alerts to return
            
        Returns:
            List[Dict[str, Any]]: List of alerts
        """
        filtered = self.alert_history
        
        if level:
            filtered = [a for a in filtered if a['level'] == level]
            
        if alert_type:
            filtered = [a for a in filtered if a['type'] == alert_type]
        
        # Return most recent alerts first
        return sorted(filtered, key=lambda x: x['timestamp'], reverse=True)[:limit]
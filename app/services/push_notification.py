"""Apple Push Notification Service (APNs) integration.

Sends push notifications to iOS devices for trading alerts.
"""

import logging
import json
import jwt
import time
import httpx
from typing import Dict, List, Optional, Literal
from datetime import datetime

from app.config import get_settings
from app.storage.device_store import get_device_store

logger = logging.getLogger(__name__)

AlertCategory = Literal["BUY", "STOP", "SELL"]


class APNsService:
    """Service for sending iOS push notifications via APNs.

    Uses JWT authentication with APNs HTTP/2 API.
    """

    # APNs endpoints
    SANDBOX_URL = "https://api.sandbox.push.apple.com"
    PRODUCTION_URL = "https://api.push.apple.com"

    def __init__(self):
        """Initialize APNs service."""
        settings = get_settings()

        self.key_id = settings.apns_key_id
        self.team_id = settings.apns_team_id
        self.key_path = settings.apns_key_path
        self.bundle_id = settings.apns_bundle_id
        self.use_sandbox = settings.apns_use_sandbox

        self._private_key: Optional[str] = None
        self._token: Optional[str] = None
        self._token_timestamp: float = 0
        self._device_store = get_device_store()

        # Token valid for 1 hour, refresh at 50 minutes
        self.TOKEN_REFRESH_INTERVAL = 50 * 60

    def _load_private_key(self) -> str:
        """Load APNs private key from file.

        Returns:
            Private key string
        """
        if self._private_key:
            return self._private_key

        if not self.key_path:
            raise ValueError("APNS_KEY_PATH not configured")

        try:
            with open(self.key_path, "r") as f:
                self._private_key = f.read()
            return self._private_key
        except FileNotFoundError:
            raise ValueError(f"APNs key file not found: {self.key_path}")

    def _generate_token(self) -> str:
        """Generate JWT token for APNs authentication.

        Returns:
            JWT token string
        """
        current_time = time.time()

        # Return cached token if still valid
        if self._token and (current_time - self._token_timestamp) < self.TOKEN_REFRESH_INTERVAL:
            return self._token

        private_key = self._load_private_key()

        headers = {
            "alg": "ES256",
            "kid": self.key_id,
        }

        payload = {
            "iss": self.team_id,
            "iat": int(current_time),
        }

        self._token = jwt.encode(
            payload,
            private_key,
            algorithm="ES256",
            headers=headers,
        )
        self._token_timestamp = current_time

        return self._token

    def _get_base_url(self) -> str:
        """Get APNs base URL based on environment.

        Returns:
            APNs endpoint URL
        """
        return self.SANDBOX_URL if self.use_sandbox else self.PRODUCTION_URL

    def _build_payload(
        self,
        alert_type: AlertCategory,
        symbol: str,
        message: str,
        price: float,
    ) -> Dict:
        """Build APNs notification payload.

        Args:
            alert_type: Type of alert (BUY, STOP, SELL)
            symbol: Stock ticker symbol
            message: Alert message
            price: Current price

        Returns:
            APNs payload dictionary
        """
        # Truncate message for notification
        short_message = message[:200] + "..." if len(message) > 200 else message

        # Set category and sound based on alert type
        if alert_type == "BUY":
            title = f"{symbol} Buy Signal"
            sound = "default"
            category = "BUY_ALERT"
        elif alert_type == "STOP":
            title = f"{symbol} Stop Alert"
            sound = "alert.wav"  # More urgent sound
            category = "STOP_ALERT"
        else:  # SELL
            title = f"{symbol} Target Hit"
            sound = "success.wav"
            category = "SELL_ALERT"

        return {
            "aps": {
                "alert": {
                    "title": title,
                    "body": short_message,
                },
                "sound": sound,
                "badge": 1,
                "category": category,
                "mutable-content": 1,  # Allows notification extension modification
            },
            "data": {
                "alert_type": alert_type,
                "symbol": symbol,
                "price": price,
                "full_message": message,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

    async def send_notification(
        self,
        device_token: str,
        alert_type: AlertCategory,
        symbol: str,
        message: str,
        price: float,
    ) -> bool:
        """Send a push notification to a single device.

        Args:
            device_token: APNs device token
            alert_type: Type of alert
            symbol: Stock ticker symbol
            message: Alert message
            price: Current price

        Returns:
            True if sent successfully, False otherwise
        """
        if not all([self.key_id, self.team_id, self.key_path, self.bundle_id]):
            logger.warning("APNs not configured - skipping push notification")
            return False

        try:
            token = self._generate_token()
            url = f"{self._get_base_url()}/3/device/{device_token}"

            headers = {
                "authorization": f"bearer {token}",
                "apns-topic": self.bundle_id,
                "apns-push-type": "alert",
                "apns-priority": "10",  # High priority
            }

            payload = self._build_payload(alert_type, symbol, message, price)

            async with httpx.AsyncClient(http2=True) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=10.0,
                )

                if response.status_code == 200:
                    logger.info(f"Push notification sent to device for {symbol} {alert_type}")
                    await self._device_store.update_last_used(device_token)
                    return True

                elif response.status_code == 410:
                    # Device token is no longer valid
                    logger.warning(f"Invalid device token, removing: {device_token[:20]}...")
                    await self._device_store.unregister_device(device_token)
                    return False

                else:
                    error_body = response.text
                    logger.error(f"APNs error {response.status_code}: {error_body}")
                    return False

        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False

    async def send_to_user(
        self,
        user_id: str,
        alert_type: AlertCategory,
        symbol: str,
        message: str,
        price: float,
    ) -> int:
        """Send push notification to all of a user's devices.

        Args:
            user_id: User identifier
            alert_type: Type of alert
            symbol: Stock ticker symbol
            message: Alert message
            price: Current price

        Returns:
            Number of successful sends
        """
        device_tokens = await self._device_store.get_device_tokens_for_user(user_id)

        if not device_tokens:
            logger.info(f"No devices registered for user {user_id}")
            return 0

        success_count = 0

        for token in device_tokens:
            if await self.send_notification(token, alert_type, symbol, message, price):
                success_count += 1

        logger.info(f"Sent {success_count}/{len(device_tokens)} push notifications for user {user_id}")
        return success_count

    async def send_batch(
        self,
        alerts: List[Dict],
    ) -> int:
        """Send multiple push notifications efficiently.

        Args:
            alerts: List of dicts with user_id, alert_type, symbol, message, price

        Returns:
            Total number of successful sends
        """
        total_success = 0

        for alert in alerts:
            count = await self.send_to_user(
                user_id=alert["user_id"],
                alert_type=alert["alert_type"],
                symbol=alert["symbol"],
                message=alert["message"],
                price=alert["price"],
            )
            total_success += count

        return total_success

    def is_configured(self) -> bool:
        """Check if APNs is properly configured.

        Returns:
            True if all required settings are present
        """
        return all([
            self.key_id,
            self.team_id,
            self.key_path,
            self.bundle_id,
        ])

    def get_status(self) -> Dict:
        """Get APNs service status.

        Returns:
            Dictionary with configuration status
        """
        return {
            "configured": self.is_configured(),
            "environment": "sandbox" if self.use_sandbox else "production",
            "bundle_id": self.bundle_id,
            "team_id": self.team_id[:4] + "..." if self.team_id else None,
            "token_valid": (
                self._token is not None and
                (time.time() - self._token_timestamp) < self.TOKEN_REFRESH_INTERVAL
            ),
        }


# Singleton instance
_apns_service: Optional[APNsService] = None


def get_apns_service() -> APNsService:
    """Get the singleton APNs service instance."""
    global _apns_service
    if _apns_service is None:
        _apns_service = APNsService()
    return _apns_service


async def send_trading_alert(
    user_id: str,
    symbol: str,
    alert_type: AlertCategory,
    message: str,
    price: float,
) -> int:
    """Convenience function to send a trading alert.

    Args:
        user_id: User identifier
        symbol: Stock ticker symbol
        alert_type: Type of alert (BUY, STOP, SELL)
        message: Alert message
        price: Current price

    Returns:
        Number of successful sends
    """
    service = get_apns_service()
    return await service.send_to_user(user_id, alert_type, symbol, message, price)

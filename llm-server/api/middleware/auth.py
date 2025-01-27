from fastapi import Request, HTTPException
from typing import Optional, Dict, Set
import json
from pathlib import Path
import logging
import time
import asyncio
from cachetools import TTLCache, LRUCache
import hashlib


class AuthMiddleware:
    def __init__(self):
        self.logger = self._setup_logging()
        self.api_keys: Dict[str, dict] = {}
        self.key_cache = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache
        self.rate_limits = LRUCache(maxsize=10000)  # Rate limiting cache
        self.blocked_ips = TTLCache(maxsize=1000, ttl=3600)  # 1-hour blocks
        self.last_reload = 0
        self._reload_lock = asyncio.Lock()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("Auth")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("/mnt/data/llm-server/logs/auth.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    async def _load_api_keys(self, model_type: str) -> dict:
        """Load API keys with error handling and caching"""
        try:
            key_path = Path(f"/mnt/data/llm-server/config/{model_type}_api_keys.json")
            if not key_path.exists():
                self.logger.error(f"API keys file not found: {key_path}")
                return {"keys": []}

            # Check if reload is needed
            stats = key_path.stat()
            if stats.st_mtime > self.last_reload:
                async with self._reload_lock:
                    with open(key_path, "r") as f:
                        keys = json.load(f)
                    self.api_keys[model_type] = keys
                    self.last_reload = stats.st_mtime
                    return keys

            return self.api_keys.get(model_type, {"keys": []})

        except Exception as e:
            self.logger.error(f"Error loading API keys for {model_type}: {e}")
            return {"keys": []}

    def _get_rate_limit_key(self, api_key: str, client_ip: str) -> str:
        """Generate unique key for rate limiting"""
        return hashlib.sha256(f"{api_key}:{client_ip}".encode()).hexdigest()

    async def _check_rate_limit(self, rate_key: str, limits: dict) -> bool:
        """Check rate limits with sliding window"""
        current_time = time.time()
        window_size = 60  # 1 minute window

        if rate_key not in self.rate_limits:
            self.rate_limits[rate_key] = []

        # Clean old requests
        self.rate_limits[rate_key] = [t for t in self.rate_limits[rate_key] if current_time - t < window_size]

        # Check limits
        if len(self.rate_limits[rate_key]) >= limits.get("requests_per_minute", 300):
            return False

        # Add new request
        self.rate_limits[rate_key].append(current_time)
        return True

    async def _validate_key(self, api_key: str, model_type: str) -> bool:
        """Validate API key with caching"""
        cache_key = f"{model_type}:{api_key}"

        # Check cache first
        if cache_key in self.key_cache:
            return self.key_cache[cache_key]

        # Load keys if needed
        keys = await self._load_api_keys(model_type)
        is_valid = api_key in keys.get("keys", [])

        # Update cache
        self.key_cache[cache_key] = is_valid
        return is_valid

    async def __call__(self, request: Request, call_next):
        """Middleware handler"""
        try:
            # Skip auth for health check
            if request.url.path == "/health":
                return await call_next(request)

            # Get client IP
            client_ip = request.client.host
            if client_ip in self.blocked_ips:
                raise HTTPException(status_code=403, detail="IP temporarily blocked")

            # Extract model type from path
            path = request.url.path
            model_type = "text" if "/v1/chat" in path else "image" if "/v1/images" in path else None

            if not model_type:
                raise HTTPException(status_code=404, detail="Invalid endpoint")

            # Get and validate API key
            api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
            if not api_key:
                raise HTTPException(status_code=401, detail="Missing API key")

            if not await self._validate_key(api_key, model_type):
                # Track failed attempts
                rate_key = self._get_rate_limit_key(api_key, client_ip)
                if rate_key not in self.rate_limits:
                    self.rate_limits[rate_key] = 1
                else:
                    self.rate_limits[rate_key] += 1

                    # Block IP if too many failed attempts
                    if self.rate_limits[rate_key] > 10:
                        self.blocked_ips[client_ip] = time.time()

                raise HTTPException(status_code=401, detail="Invalid API key")

            # Check rate limits
            rate_key = self._get_rate_limit_key(api_key, client_ip)
            if not await self._check_rate_limit(rate_key, self.api_keys[model_type]):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            # Add request ID for tracking
            request.state.request_id = hashlib.sha256(f"{time.time()}:{api_key}:{client_ip}".encode()).hexdigest()[:16]

            response = await call_next(request)
            return response

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            raise HTTPException(status_code=500, detail="Internal authentication error")

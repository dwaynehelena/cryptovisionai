import time
import logging
from typing import Dict, Tuple, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class CacheEntry:
    def __init__(self, content: bytes, headers: dict, ttl: int):
        self.content = content
        self.headers = headers
        self.expires_at = time.time() + ttl

class InMemoryCache:
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = 1000

    def get(self, key: str) -> Optional[CacheEntry]:
        entry = self._cache.get(key)
        if not entry:
            return None
        
        if time.time() > entry.expires_at:
            del self._cache[key]
            return None
            
        return entry

    def set(self, key: str, content: bytes, headers: dict, ttl: int):
        # Simple eviction if full
        if len(self._cache) >= self._max_size:
            # Remove oldest 10%
            keys_to_remove = list(self._cache.keys())[:int(self._max_size * 0.1)]
            for k in keys_to_remove:
                del self._cache[k]
                
        self._cache[key] = CacheEntry(content, headers, ttl)

    def clear(self):
        self._cache.clear()

# Global cache instance
cache_store = InMemoryCache()

class CacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, ttl: int = 60):
        super().__init__(app)
        self.ttl = ttl
        self.skip_paths = [
            "/health",
            "/api/v1/orders",
            "/api/v1/risk",
            "/ws"
        ]

    async def dispatch(self, request: Request, call_next):
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)

        # Check skip paths
        path = request.url.path
        for skip in self.skip_paths:
            if path.startswith(skip):
                return await call_next(request)

        # Generate cache key
        cache_key = f"{request.method}:{request.url}"
        
        # Check cache
        cached = cache_store.get(cache_key)
        if cached:
            logger.debug(f"Cache hit: {cache_key}")
            return Response(
                content=cached.content,
                headers=dict(cached.headers),
                media_type=cached.headers.get("content-type")
            )

        # Execute request
        response = await call_next(request)

        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            response_body = [section async for section in response.body_iterator]
            content = b"".join(response_body)
            
            # Store in cache
            # Determine TTL based on path
            ttl = self.ttl
            if "/portfolio" in path:
                ttl = 10 # Shorter TTL for portfolio
            elif "/market" in path:
                ttl = 30 # Medium TTL for market data
            
            cache_store.set(cache_key, content, dict(response.headers), ttl)
            
            # Create new response to avoid Content-Length mismatch
            headers = dict(response.headers)
            headers.pop("content-length", None) # Let Starlette recalculate
            
            return Response(
                content=content,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
                background=response.background
            )

        return response

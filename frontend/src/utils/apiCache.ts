/**
 * API Response Cache
 * Simple in-memory cache with TTL and LRU eviction policy
 */

interface CacheEntry<T> {
    data: T;
    timestamp: number;
    ttl: number;
}

class ApiCache {
    private cache: Map<string, CacheEntry<any>> = new Map();
    private maxEntries: number;
    private defaultTTL: number;

    constructor(maxEntries: number = 100, defaultTTL: number = 60000) {
        this.maxEntries = maxEntries;
        this.defaultTTL = defaultTTL;
    }

    /**
     * Generate a cache key from URL and params
     */
    generateKey(url: string, params?: any): string {
        if (!params) return url;
        // Sort params to ensure consistent keys
        const sortedParams = Object.keys(params)
            .sort()
            .reduce((acc: any, key) => {
                acc[key] = params[key];
                return acc;
            }, {});
        return `${url}?${JSON.stringify(sortedParams)}`;
    }

    /**
     * Get data from cache
     */
    get<T>(key: string): T | null {
        const entry = this.cache.get(key);

        if (!entry) return null;

        // Check expiration
        if (Date.now() - entry.timestamp > entry.ttl) {
            this.cache.delete(key);
            return null;
        }

        // Refresh LRU position (delete and re-add)
        this.cache.delete(key);
        this.cache.set(key, entry);

        return entry.data;
    }

    /**
     * Set data in cache
     */
    set<T>(key: string, data: T, ttl: number = this.defaultTTL): void {
        // Evict oldest if full
        if (this.cache.size >= this.maxEntries) {
            const firstKey = this.cache.keys().next().value;
            if (firstKey) this.cache.delete(firstKey);
        }

        this.cache.set(key, {
            data,
            timestamp: Date.now(),
            ttl
        });
    }

    /**
     * Clear cache
     */
    clear(): void {
        this.cache.clear();
    }

    /**
     * Remove specific entry
     */
    remove(key: string): void {
        this.cache.delete(key);
    }

    /**
     * Invalidate all entries matching a pattern
     */
    invalidatePattern(pattern: RegExp): void {
        for (const key of this.cache.keys()) {
            if (pattern.test(key)) {
                this.cache.delete(key);
            }
        }
    }
}

export const apiCache = new ApiCache();
export default apiCache;

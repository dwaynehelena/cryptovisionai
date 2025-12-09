/**
 * Request Queue Manager
 * Limits concurrent API requests to prevent backend overload
 */

interface QueuedRequest {
    id: string;
    execute: () => Promise<any>;
    resolve: (value: any) => void;
    reject: (error: any) => void;
    priority: number;
}

class RequestQueue {
    private queue: QueuedRequest[] = [];
    private activeRequests = 0;
    private maxConcurrent: number;
    private requestIdCounter = 0;

    constructor(maxConcurrent: number = 5) {
        this.maxConcurrent = maxConcurrent;
    }

    /**
     * Add a request to the queue
     * @param execute Function that executes the request
     * @param priority Higher priority requests are executed first (default: 0)
     */
    async enqueue<T>(
        execute: () => Promise<T>,
        priority: number = 0
    ): Promise<T> {
        return new Promise((resolve, reject) => {
            const request: QueuedRequest = {
                id: `req_${++this.requestIdCounter}`,
                execute,
                resolve,
                reject,
                priority,
            };

            // Insert request in priority order
            const insertIndex = this.queue.findIndex((r) => r.priority < priority);
            if (insertIndex === -1) {
                this.queue.push(request);
            } else {
                this.queue.splice(insertIndex, 0, request);
            }

            this.processQueue();
        });
    }

    private async processQueue() {
        // Process requests if we have capacity
        while (this.activeRequests < this.maxConcurrent && this.queue.length > 0) {
            const request = this.queue.shift();
            if (!request) break;

            this.activeRequests++;

            // Execute request
            request
                .execute()
                .then((result) => {
                    request.resolve(result);
                })
                .catch((error) => {
                    request.reject(error);
                })
                .finally(() => {
                    this.activeRequests--;
                    this.processQueue(); // Process next request
                });
        }
    }

    /**
     * Get current queue statistics
     */
    getStats() {
        return {
            queued: this.queue.length,
            active: this.activeRequests,
            maxConcurrent: this.maxConcurrent,
        };
    }

    /**
     * Update max concurrent requests
     */
    setMaxConcurrent(max: number) {
        this.maxConcurrent = max;
        this.processQueue();
    }

    /**
     * Clear all pending requests
     */
    clear() {
        this.queue.forEach((req) => {
            req.reject(new Error('Request queue cleared'));
        });
        this.queue = [];
    }
}

// Create singleton instance
export const requestQueue = new RequestQueue(5);

// Export class for testing
export { RequestQueue };

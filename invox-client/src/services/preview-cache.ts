import { getArtifactPreviewPicture } from "./artifacts";

interface CacheEntry {
	url: string;
	timestamp: number;
}

class PreviewCacheService {
	private cache = new Map<string, CacheEntry>();
	private readonly CACHE_DURATION = 30 * 60 * 1000; // 30 minutes
	private readonly MAX_CACHE_SIZE = 100; // Maximum number of cached images

	/**
	 * Get a preview image URL from cache or fetch it
	 */
	async getPreviewUrl(documentId: string): Promise<string> {
		// Check if we have a valid cached entry
		const cached = this.cache.get(documentId);
		if (cached && this.isValidCacheEntry(cached)) {
			return cached.url;
		}

		// Clean up old cache entry if it exists
		if (cached) {
			this.revokeUrl(cached.url);
			this.cache.delete(documentId);
		}

		// Fetch new image
		try {
			const blob = await getArtifactPreviewPicture(documentId);
			const url = window.URL.createObjectURL(blob);
			
			// Store in cache
			this.cache.set(documentId, {
				url,
				timestamp: Date.now()
			});

			// Clean up old entries if cache is getting too large
			this.cleanupCache();

			return url;
		} catch (error) {
			console.error(`Failed to fetch preview for document ${documentId}:`, error);
			throw error;
		}
	}

	/**
	 * Prefetch preview images for multiple documents
	 */
	async prefetchPreviews(documentIds: string[]): Promise<void> {
		const uncachedIds = documentIds.filter(id => {
			const cached = this.cache.get(id);
			return !cached || !this.isValidCacheEntry(cached);
		});

		if (uncachedIds.length === 0) return;

		// Fetch in batches to avoid overwhelming the server
		const BATCH_SIZE = 5;
		for (let i = 0; i < uncachedIds.length; i += BATCH_SIZE) {
			const batch = uncachedIds.slice(i, i + BATCH_SIZE);
			
			await Promise.allSettled(
				batch.map(async (documentId) => {
					try {
						await this.getPreviewUrl(documentId);
					} catch (error) {
						// Silently fail for prefetch - individual components will handle errors
						console.warn(`Failed to prefetch preview for ${documentId}:`, error);
					}
				})
			);

			// Small delay between batches to be nice to the server
			if (i + BATCH_SIZE < uncachedIds.length) {
				await new Promise(resolve => setTimeout(resolve, 100));
			}
		}
	}

	/**
	 * Check if a cached URL is available (not revoked)
	 */
	isUrlAvailable(documentId: string): boolean {
		const cached = this.cache.get(documentId);
		return cached ? this.isValidCacheEntry(cached) : false;
	}

	/**
	 * Get cached URL without fetching (returns null if not available)
	 */
	getCachedUrl(documentId: string): string | null {
		const cached = this.cache.get(documentId);
		if (cached && this.isValidCacheEntry(cached)) {
			return cached.url;
		}
		return null;
	}

	/**
	 * Clear all cached images
	 */
	clearCache(): void {
		for (const entry of this.cache.values()) {
			this.revokeUrl(entry.url);
		}
		this.cache.clear();
	}

	/**
	 * Remove a specific document from cache
	 */
	removeFromCache(documentId: string): void {
		const cached = this.cache.get(documentId);
		if (cached) {
			this.revokeUrl(cached.url);
			this.cache.delete(documentId);
		}
	}

	private isValidCacheEntry(entry: CacheEntry): boolean {
		return Date.now() - entry.timestamp < this.CACHE_DURATION;
	}

	private cleanupCache(): void {
		if (this.cache.size <= this.MAX_CACHE_SIZE) return;

		// Remove expired entries first
		const now = Date.now();
		for (const [documentId, entry] of this.cache.entries()) {
			if (!this.isValidCacheEntry(entry)) {
				this.revokeUrl(entry.url);
				this.cache.delete(documentId);
			}
		}

		// If still too large, remove oldest entries
		if (this.cache.size > this.MAX_CACHE_SIZE) {
			const entries = Array.from(this.cache.entries());
			entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
			
			const toRemove = entries.slice(0, entries.length - this.MAX_CACHE_SIZE);
			for (const [documentId, entry] of toRemove) {
				this.revokeUrl(entry.url);
				this.cache.delete(documentId);
			}
		}
	}

	private revokeUrl(url: string): void {
		try {
			window.URL.revokeObjectURL(url);
		} catch (error) {
			// Ignore errors when revoking URLs
		}
	}
}

export const previewCache = new PreviewCacheService(); 
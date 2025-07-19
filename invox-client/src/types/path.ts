/**
 * Represents a single item (file or folder) within a path listing.
 */
export interface PathContentItem {
	name: string;
	path: string;
	type: "file" | "folder";
	serviceId: number;
}

/**
 * Service metadata for display purposes
 */
export interface ServiceMetadata {
	serviceName?: string;
	host?: string;
	serviceProvider?: string;
}

/**
 * The result structure returned by the path content retrieval function.
 */
export interface ServicePathContentsResult {
	serviceId: number;
	items: PathContentItem[];
	serviceMetadata?: ServiceMetadata;
}

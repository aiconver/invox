export interface UnifiedResource {
	id: string;
	dbId: number;
	displayName: string;
	isEnabled: boolean;
	status: string; // e.g., 'idle', 'crawling', 'error_permanent'
	createdAt: Date;
}

// A type for the credentials that combines the v1 and v2 credentials
export interface UnifiedCredential {
	id: string; // A unique ID for the list in the frontend, combining type and db id
	idmUserId: string;
	version: '1' | '2';
	provider: string; // 'webdav', 'samba', 'ms_graph'
	displayName: string;
	status?: string;
	createdAt: Date;
	// V1-specific ID
	credentialId?: number;
	// V2-specific ID and nested resources
	serviceAuthId?: string;
	resources?: UnifiedResource[];
}

// Keep the old ServiceCredential interface for backward compatibility during transition
export interface ServiceCredential {
	id: string;
	alias?: string;
	service: ServiceType;
	user: string;
	pass: string;
	host: string;
	lastChangeAt: string;
	isCrawling: boolean;
	settings: string;
}

export const SERVICE_TYPES = ["samba", "webdav", "ms_graph"] as const;

export type ServiceType = (typeof SERVICE_TYPES)[number];

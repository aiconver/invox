export interface AuthorProfile {
	id: string;
	name: string;
	avatar?: string | null;
}

export interface PreviewItem {
	topicId: number;
	colorId: number;
	similarity: number;
	text: string;
}

export interface Summary {
	id: number;
	colorId: number;
	topWords: any;
}

export interface Topics {
	[id: number]: number;
}

export interface Artifact {
	id: string;
	name: string;
	originalId: string;
	originalPath: string;
	pageCount: number;
	authors: AuthorProfile[];
	owners: AuthorProfile[];
	summary: Summary[];
	preview: PreviewItem[];
	modification: string;
	creation: string;
	serviceId: string;
	serviceName: string;
	type: string;
	mimeType?: string;
	metadata: string;
	parent: string;
	topics: Topics;
	tokenCount: number;
	readingTimeMinutes: number;
	language: string;
	summaryShort: string;
	summaryLong: string;
	summaryShortGenerated: string;
	summaryLongGenerated: string;
	translations: { language: string; content: string }[];
	embedding: number[];
	chunks: any[];
	content: string[];
	relevanceScore?: number;
}

export interface ArtifactSearchResult {
	artifacts: Artifact[];
	metadata: {
		artifacts: {
			totalCount: number;
			hasMore: boolean;
		};
	};
}

/* Filters */

const FILE_CATEGORIES = ["Documents", "Presentations", "Images", "Videos", "Audio", "Archives", "Data"] as const

export type FileCategories = typeof FILE_CATEGORIES[number]

export interface ArtifactFilters {
	fileTypes?: FileCategories[];
}

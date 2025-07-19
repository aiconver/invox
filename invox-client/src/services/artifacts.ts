import { apiClient } from "@/lib/axios";
import type { ArtifactSearchResult, ArtifactFilters } from "@/types/artifacts";
import { FILE_TYPE_CATEGORIES } from "@/components/artifacts/v1/artifact-filters";

export const getArtifacts = async (params: {
	text?: string;
	offset?: number;
	size?: number;
	filters?: ArtifactFilters;
}  = {}): Promise<ArtifactSearchResult> => {
	const searchParams = new URLSearchParams();
	
	if (params.text) {
		searchParams.append("text", params.text);
	}
	if (params.offset !== undefined) {
		searchParams.append("offset", params.offset.toString());
	}
	if (params.size !== undefined) {
		searchParams.append("size", params.size.toString());
	}

	const response = await apiClient.get(`/api/v2/artifacts?${searchParams.toString()}`);
	
	let result = response.data as ArtifactSearchResult;
	
	const activeMimeTypes = params.filters?.fileTypes?.flatMap(category => FILE_TYPE_CATEGORIES[category] || []) || [];

	if (activeMimeTypes.length > 0) {
		result = {
			...result,
			artifacts: result.artifacts.filter(artifact => 
				activeMimeTypes.includes(artifact.mimeType || "")
			),
		};
	}
	
	if (activeMimeTypes.length > 0) {
		result = {
			...result,
			metadata: {
				...result.metadata,
				artifacts: {
					...result.metadata.artifacts,
					totalCount: result.artifacts.length,
					hasMore: false,
				}
			}
		};
	}
	
	return result;
};

export const getArtifactsV1 = async (params: {
	text?: string;
	offset?: number;
	size?: number;
}  = {}): Promise<ArtifactSearchResult> => {
	const searchParams = new URLSearchParams();
	
	if (params.text) {
		searchParams.append("text", params.text);
	}
	if (params.offset !== undefined) {
		searchParams.append("offset", params.offset.toString());
	}
	if (params.size !== undefined) {
		searchParams.append("size", params.size.toString());
	}

	const response = await apiClient.get(`/api/v1/artifacts?${searchParams.toString()}`);
	
	return response.data as ArtifactSearchResult;
};

export const getArtifactPdf = async (documentId: string, chunkIndex?: string): Promise<Blob> => {
	const response = await apiClient.get(
		`/api/v1/artifacts/pdf/${documentId}/pdf-content?${
			chunkIndex ? `chunkIndex=${chunkIndex}` : ""
		}`,
		{ responseType: "blob" },
	);
	return response.data;
};

export const getArtifactPreviewPicture = async (documentId: string): Promise<Blob> => {
	const response = await apiClient.get(`/api/v1/artifacts/${documentId}/preview-picture`, {
		responseType: 'blob'
	});
	return response.data;
}; 
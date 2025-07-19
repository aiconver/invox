import { apiClient } from "@/lib/axios";
import type { DiscoveredResource } from "@/types/ms-graph";

export async function fetchDiscoverableResources(
	serviceAuthId: string,
): Promise<DiscoveredResource[]> {
	const response = await apiClient.get<DiscoveredResource[]>(
		`/api/v1/credentials/ms-graph/discoverable-resources/${serviceAuthId}`,
	);
	return response.data || [];
}

export async function connectMsGraphResources(payload: {
	serviceAuthId: string;
	selectedResources: DiscoveredResource[];
}) {
	const response = await apiClient.post("/api/v1/credentials/ms-graph/connect-resources", payload);
	return response.data;
}

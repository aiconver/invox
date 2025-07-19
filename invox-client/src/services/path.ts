import { apiClient } from "@/lib/axios";
import type { ServicePathContentsResult } from "@/types/path";

export const fetchPathContents = async (path: string) => {
	const response = await apiClient.get("/api/v1/artifacts/paths/contents", {
		params: {
			path: path,
		},
	});

	console.log("Path contents response", response.data);
	return response.data as ServicePathContentsResult[];
};

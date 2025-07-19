import { apiClient } from "@/lib/axios";
import type { ServiceCredential, UnifiedCredential } from "@/types/credentials";

export const getCredentials = async () => {
	const response = await apiClient.get("/api/v2/credentials");

	console.log("Credentials response", response.data);
	return response.data as UnifiedCredential[];
};

export const deleteCredential = async (id: string) => {
	const response = await apiClient.delete(`/api/v2/credentials/${id}`);
	console.log("Credential deleted", response.data);
	return response.data;

};

export const createCredential = async (credential: {
	host: string;
	user: string;
	pass: string;
	service: string;
	settings?: string;
}) => {
	const response = await apiClient.put("/api/v1/credentials", {
		credentials: credential,
	});
	return response.data;
};

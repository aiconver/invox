import { getUser } from "@/lib/auth";
import axios from "axios";

const apiClient = axios.create();

apiClient.interceptors.request.use(
	(config) => {
		const user = getUser();
		const token = user?.access_token;
		if (token) {
			config.headers.Authorization = `Bearer ${token}`;
		}
		return config;
	},
	(error) => {
		return Promise.reject(error);
	},
);

export { apiClient };

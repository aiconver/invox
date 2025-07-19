import { User } from "oidc-client-ts";

export function getUser() {
	const oidcStorage = localStorage.getItem(
		`oidc.user:${import.meta.env.VITE_APP_KEYCLOAK_URL}realms/${import.meta.env.VITE_APP_KEYCLOAK_REALM}:${import.meta.env.VITE_APP_KEYCLOAK_CLIENTID}`,
	);
	if (!oidcStorage) {
		return null;
	}

	return User.fromStorageString(oidcStorage);
}

export function getHeaders() {
	const user = getUser();
	const token = user?.access_token;
	return {
		Authorization: `Bearer ${token}`,
	};
}

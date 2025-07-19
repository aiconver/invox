import { UserManager, WebStorageStateStore } from "oidc-client-ts";
import type { ReactNode } from "react";
import { AuthProvider as OidcAuthProvider } from "react-oidc-context";

export const userManager = new UserManager({
	authority: `${import.meta.env.VITE_APP_KEYCLOAK_URL}realms/${import.meta.env.VITE_APP_KEYCLOAK_REALM}`,
	// biome-ignore lint/style/useNamingConvention: Naming comes from library
	client_id: import.meta.env.VITE_APP_KEYCLOAK_CLIENTID,
	// biome-ignore lint/style/useNamingConvention: Naming comes from library
	redirect_uri: `${window.location.origin}${window.location.pathname}`,
	// biome-ignore lint/style/useNamingConvention: Naming comes from library
	post_logout_redirect_uri: window.location.origin,
	userStore: new WebStorageStateStore({ store: window.localStorage }),
	monitorSession: true, // this allows cross tab login/logout detection
	//loadUserInfo: true,
});

export const onSigninCallback = () => {
	window.history.replaceState({}, document.title, window.location.pathname);
};

export function AuthProvider({ children }: { children: ReactNode }) {
	return (
		<OidcAuthProvider
			userManager={userManager}
			onSigninCallback={onSigninCallback}
		>
			{children}
		</OidcAuthProvider>
	);
}

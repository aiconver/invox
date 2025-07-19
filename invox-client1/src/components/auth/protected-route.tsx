import { LayoutSkeleton } from "@/components/skeleton/layout-skeleton";

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { hasAuthParams, useAuth } from "react-oidc-context";
import { Outlet } from "react-router-dom";
import { AuthError } from "./auth-error";

export function ProtectedRoute() {
	// const auth = useAuth();
	// const { t } = useTranslation();
	// const [hasTriedSignin, setHasTriedSignin] = useState(false);

	// useEffect(() => {
	// 	if (
	// 		!(
	// 			hasAuthParams() ||
	// 			auth.isAuthenticated ||
	// 			auth.activeNavigator ||
	// 			auth.isLoading ||
	// 			hasTriedSignin
	// 		)
	// 	) {
	// 		void auth.signinRedirect();
	// 		setHasTriedSignin(true);
	// 	}
	// }, [auth, hasTriedSignin]);

	// if (auth.error) {
	// 	return (
	// 		<AuthError
	// 			title={t("auth.error.signinTrouble.title")}
	// 			description={t("auth.error.signinTrouble.description")}
	// 		/>
	// 	);
	// }

	// if (auth.isLoading) {
	// 	return <LayoutSkeleton />;
	// }

	// if (!auth.isAuthenticated) {
	// 	return (
	// 		<AuthError
	// 			title={t("auth.error.notAuthenticated.title")}
	// 			description={t("auth.error.notAuthenticated.description")}
	// 		/>
	// 	);
	// }


	return <Outlet />;
}

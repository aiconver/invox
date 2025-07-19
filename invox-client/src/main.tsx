import "@/styles/global.css";
import "./i18n.ts";

import { AuthProvider } from "@/components/auth/auth-provider.tsx";
import { MainLayout } from "@/components/layout/main-layout.tsx";
import { QuestionsAnswersPage } from "@/pages/questions-answers.tsx";
import { NotFound } from "@/pages/not-found.tsx";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { APP_ROUTES } from "@/lib/routes.tsx";
import { ProtectedRoute } from "./components/auth/protected-route.tsx";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { CredentialsListPage } from "@/pages/settings/credentials-list.tsx";
import { CredentialsNewSelectPage } from "@/pages/settings/credentials-new-select.tsx";
import { CredentialsAddPage } from "@/pages/settings/credentials-add.tsx";
import { CredentialsMsAuthPage } from "@/pages/settings/credentials-ms-auth.tsx";
import { CredentialsMsGraphSelectResourcesPage } from "@/pages/settings/credentials-ms-graph-select-resources.tsx";
import { CredentialsErrorPage } from "./pages/settings/credentials-error.tsx";
import { ArtifactsPage } from "./pages/artifacts.tsx";
import { About } from "./pages/about.tsx";

const rootElement = document.getElementById("root");
if (!rootElement) {
	throw new Error("Root element not found");
}

const queryClient = new QueryClient();

const root = createRoot(rootElement);
root.render(
	<StrictMode>
		<QueryClientProvider client={queryClient}>
			<TooltipProvider>
				<BrowserRouter>
					{/* <AuthProvider> */}
						<Toaster />
						<Routes>
							{/* Main routes */}
							<Route element={<ProtectedRoute />}>
								<Route element={<MainLayout />}>
									{/* <Route
										path={APP_ROUTES["questions-answers"].to}
										element={<QuestionsAnswersPage />}
									/> */}
									<Route
										path={APP_ROUTES["artifacts"].to}
										element={<ArtifactsPage />}
									/>
									<Route
										path={APP_ROUTES["about"].to}
										element={<About />}
									/>
									{/* Settings credentials routes */}
									<Route
										path={APP_ROUTES.settings.to}
										element={
											<Navigate
												to={APP_ROUTES["settings-credentials-list"].to}
												replace
											/>
										}
									/>
									<Route
										path={APP_ROUTES["settings-credentials-list"].to}
										element={<CredentialsListPage />}
									/>
									<Route
										path={APP_ROUTES["settings-credentials-new-select"].to}
										element={<CredentialsNewSelectPage />}
									/>
									<Route
										path={`${APP_ROUTES["settings-credentials-new-form"].to}/:serviceType`}
										element={<CredentialsAddPage />}
									/>
									<Route
										path={APP_ROUTES["settings-credentials-new-ms-auth"].to}
										element={<CredentialsMsAuthPage />}
									/>
									<Route
										path={
											APP_ROUTES["settings-credentials-ms-graph-select-resources"]
												.to
										}
										element={<CredentialsMsGraphSelectResourcesPage />}
									/>
									<Route
										path={APP_ROUTES["settings-credentials-error"].to}
										element={<CredentialsErrorPage />}
									/>
								</Route>
							</Route>

							{/* Not found */}
							<Route path="*" element={<NotFound />} />
						</Routes>
					{/* </AuthProvider> */}
				</BrowserRouter>
			</TooltipProvider>
		</QueryClientProvider>
	</StrictMode>,
);

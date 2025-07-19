import { Navbar } from "@/components/layout/navbar";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";

export function CredentialsErrorPage() {
	const [searchParams] = useSearchParams();
	const error = searchParams.get("error");

	const errorMessage = useMemo(() => {
		switch (error) {
			case "msoauth_missing_state":
				return "Missing state parameter";
			case "msoauth_session_error":
				return "Session error";
			case "msoauth_invalid_state":
				return "Invalid state parameter";
			case "msoauth_code_missing":
				return "Missing auth code";
			case "msoauth_token_retrieval":
				return "Token retrieval error";
			case "msoauth_profile_fetch":
				return "Profile fetch error";
			case "msoauth_internal_error":
				return "Internal error";
			default:
				return "Unknown error";
		}
	}, [error]);

	return (
		<main className="flex flex-col h-full">
			<Navbar />
			<div className="p-8">
				<Card className="p-4">
					<Alert variant="destructive">
						<AlertTitle>Error</AlertTitle>
						<AlertDescription>{errorMessage}</AlertDescription>
					</Alert>
				</Card>
			</div>
		</main>
	);
}

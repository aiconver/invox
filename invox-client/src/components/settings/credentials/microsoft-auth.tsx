import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { MicrosoftIcon } from "@/components/icons/microsoft";
import { ExternalLink } from "lucide-react";
import { Link } from "react-router-dom";
import { APP_ROUTES } from "@/lib/routes";

export function MicrosoftAuth() {
	const { t } = useTranslation();

	const onMsAuthStart = () => {
		window.location.href = "/api/v1/credentials/oauth/ms-graph/authorize";
	};

	return (
		<div className="flex flex-col h-full p-8 gap-4">
			<div className="flex justify-end items-center">
				<Button asChild>
					<Link to={APP_ROUTES["settings-credentials-new-select"].to}>
						{t("common.back", "Back")}
					</Link>
				</Button>
			</div>

			<Card>
				<CardHeader className="text-center">
					<div className="mx-auto mb-4 w-16 h-16 flex items-center justify-center">
						<MicrosoftIcon className="w-12 h-12" />
					</div>
					<CardTitle className="text-xl">
						{t(
							"components.settings.credentials.microsoft.title",
							"Microsoft 365 Authentication",
						)}
					</CardTitle>
					<CardDescription>
						{t(
							"components.settings.credentials.microsoft.subtitle",
							"Connect your Microsoft 365 account",
						)}
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="text-center space-y-4">
						<p className="text-muted-foreground">
							{t(
								"components.settings.credentials.microsoft.description",
								"Microsoft 365 authentication requires a different process than traditional username/password credentials. Click the button below to start the OAuth authentication flow.",
							)}
						</p>
						<Button size="lg" onClick={onMsAuthStart}>
							<ExternalLink className="w-4 h-4 mr-2" />
							{t(
								"components.settings.credentials.microsoft.start_auth",
								"Start Microsoft Authentication",
							)}
						</Button>
						<p className="text-xs text-muted-foreground">
							{t(
								"components.settings.credentials.microsoft.note",
								"You will be redirected to Microsoft's login page to complete the authentication process.",
							)}
						</p>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}

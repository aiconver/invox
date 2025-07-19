import { useTranslation } from "react-i18next";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { serviceConfigs } from "@/lib/service-config";
import { Link, useNavigate } from "react-router-dom";
import { APP_ROUTES } from "@/lib/routes";
import { Button } from "@/components/ui/button";

export function CredentialsSelect() {
	const { t } = useTranslation();
	const navigate = useNavigate();

	const handleServiceSelect = (serviceKey: string) => {
		if (serviceKey === "ms_graph") {
			navigate(APP_ROUTES["settings-credentials-new-ms-auth"].to);
		} else {
			navigate(
				`${APP_ROUTES["settings-credentials-new-form"].to}/${serviceKey}`,
			);
		}
	};

	return (
		<div className="flex flex-col h-full">
			<div className="flex-1 overflow-y-auto p-6">
				<div className="mb-4 flex justify-between items-center gap-4">
					<div className="flex flex-col">
						<h2 className="text-xl font-semibold mb-2">
							{t(
								"components.settings.credentials.select.title",
								"Select Service Type",
							)}
						</h2>
						<p className="text-muted-foreground">
							{t(
								"components.settings.credentials.select.description",
								"Choose the type of service you want to connect to",
							)}
						</p>
					</div>
					<Button asChild>
						<Link to={APP_ROUTES["settings-credentials-list"].to}>
							{t("common.back")}
						</Link>
					</Button>
				</div>
				<div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
					{Object.values(serviceConfigs).map((service) => (
						<Card
							key={service.key}
							className="cursor-pointer hover:shadow-md transition-shadow"
							onClick={() => handleServiceSelect(service.key)}
						>
							<CardHeader className="text-center">
								<div className="mx-auto mb-2 w-12 h-12 flex items-center justify-center">
									<service.icon className="w-8 h-8 text-primary" />
								</div>
								<CardTitle className="text-lg">{service.title}</CardTitle>
							</CardHeader>
							<CardContent className="text-center">
								<CardDescription>{service.description}</CardDescription>
							</CardContent>
						</Card>
					))}
				</div>
			</div>
		</div>
	);
}

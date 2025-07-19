import { useTranslation } from "react-i18next";
import { useSearchParams, useNavigate, useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { APP_ROUTES } from "@/lib/routes";
import {
	Card,
	CardContent,
	CardHeader,
	CardTitle,
	CardDescription,
} from "@/components/ui/card";
import { Loader2 } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { DiscoveredResource } from "@/types/ms-graph";
import { connectMsGraphResources } from "@/services/ms-graph";
import { fetchDiscoverableResources } from "@/services/ms-graph";

export function MsGraphSelectResources() {
	const { t } = useTranslation();
	const [searchParams, setSearchParams] = useSearchParams();
	const navigate = useNavigate();
	const { serviceAuthId } = useParams<{ serviceAuthId: string }>();
	const queryClient = useQueryClient();
	const [showSuccess, setShowSuccess] = useState(false);
	const [selectedResources, setSelectedResources] = useState<
		DiscoveredResource[]
	>([]);

	const newCredential = searchParams.get("newCredential");

	useEffect(() => {
		if (newCredential === "true") {
			setShowSuccess(true);
			const newSearchParams = new URLSearchParams(searchParams);
			newSearchParams.delete("newCredential");
			setSearchParams(newSearchParams, { replace: true });
		}
	}, [newCredential, searchParams, setSearchParams]);

	const {
		data: resources,
		isLoading: isLoadingResources,
		error: fetchError,
	} = useQuery<DiscoveredResource[], Error>({
		queryKey: ["msgraph-discoverable-resources", serviceAuthId],
		queryFn: () => {
			if (!serviceAuthId) {
				throw new Error("Service Authentication ID is required.");
			}
			return fetchDiscoverableResources(serviceAuthId);
		},
		enabled: !!serviceAuthId,
	});

	console.log("resources", resources);

	const connectMutation = useMutation({
		mutationFn: connectMsGraphResources,
		onSuccess: () => {
			toast.success(t("common.success"));
			queryClient.invalidateQueries({ queryKey: ["credentials"] });
			queryClient.invalidateQueries({
				queryKey: ["msgraph-discoverable-resources", serviceAuthId],
			});
			navigate(APP_ROUTES["settings-credentials-list"].to, { replace: true });
		},
		onError: (error: Error) => {
			console.error("Error connecting resources: ", error);
			toast.error(`${t("common.error")}: ${error.message}`);
		},
	});

	const handleResourceToggle = (resource: DiscoveredResource) => {
		setSelectedResources((prev) =>
			prev.find((r) => r.id === resource.id && r.type === resource.type)
				? prev.filter(
						(r) => !(r.id === resource.id && r.type === resource.type),
					)
				: [...prev, resource],
		);
	};

	const handleSubmit = () => {
		if (!serviceAuthId || selectedResources.length === 0) {
			return;
		}
		connectMutation.mutate({ serviceAuthId, selectedResources });
	};

	if (isLoadingResources) {
		return (
			<div className="flex flex-col items-center justify-center h-full">
				<Loader2 className="w-12 h-12 animate-spin text-primary" />
			</div>
		);
	}

	if (fetchError) {
		return (
			<div className="flex flex-col items-center justify-center h-full p-8">
				<Card className="w-full max-w-md">
					<CardHeader>
						<CardTitle className="text-destructive">
							{t("common.error")}
						</CardTitle>
					</CardHeader>
					<CardContent>
						<p className="text-sm text-muted-foreground mt-2">
							{fetchError.message ? fetchError.message : t("common.error")}
						</p>
					</CardContent>
				</Card>
			</div>
		);
	}

	return (
		<div className="p-8 flex flex-col h-full">
			{showSuccess && (
				<div
					className="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 mb-6 rounded-md"
					role="alert"
				>
					<p className="font-bold">
						{t(
							"components.settings.credentials.msGraphSelect.authSuccessTitle",
						)}
					</p>
					<p>
						{t(
							"components.settings.credentials.msGraphSelect.authSuccessMessage",
						)}
					</p>
				</div>
			)}

			<Card className="flex-1 flex flex-col">
				<CardHeader>
					<CardTitle>
						{t("components.settings.credentials.msGraphSelect.title")}
					</CardTitle>
					<CardDescription>
						{t("components.settings.credentials.msGraphSelect.description")}
					</CardDescription>
				</CardHeader>
				<CardContent className="flex-1 overflow-y-auto pr-2 space-y-4">
					{!resources || resources.length === 0 ? (
						<p className="text-center text-muted-foreground py-8">
							{t(
								"components.settings.credentials.msGraphSelect.noResourcesFound",
							)}
						</p>
					) : (
						resources.map((resource) => (
							<div
								key={`${resource.type}-${resource.id}`}
								className="flex items-center justify-between p-3 border rounded-md"
							>
								<div>
									<p className="font-medium">{resource.name}</p>
									<p className="text-sm text-muted-foreground">
										{resource.type === "onedrive"
											? "OneDrive"
											: "SharePoint"}
									</p>
								</div>
								<Button
									variant={
										selectedResources.find(
											(r) => r.id === resource.id && r.type === resource.type,
										)
											? "default"
											: "outline"
									}
									onClick={() => handleResourceToggle(resource)}
									disabled={connectMutation.isPending}
								>
									{selectedResources.find(
										(r) => r.id === resource.id && r.type === resource.type,
									)
										? t("common.selected")
										: t("common.select")}
								</Button>
							</div>
						))
					)}
				</CardContent>
			</Card>
			<div className="mt-6 flex justify-end gap-2">
				<Button
					onClick={handleSubmit}
					disabled={
						isLoadingResources ||
						selectedResources.length === 0 ||
						connectMutation.isPending
					}
				>
					{connectMutation.isPending ? (
						<Loader2 className="w-4 h-4 mr-2 animate-spin" />
					) : null}
					{t("common.save")}
				</Button>
			</div>
		</div>
	);
}

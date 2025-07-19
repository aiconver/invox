import { Button } from "@/components/ui/button";

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { toast } from "sonner";
import { Loader2, Trash2, ChevronDown, ChevronRight, Eye, EyeOff } from "lucide-react";
import { getCredentials, deleteCredential } from "@/services/credentials";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { UnifiedCredential } from "@/types/credentials";
import { useMemo, useState } from "react";
import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
	AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { getServiceIcon, getServiceLabel } from "@/lib/service-config";
import { APP_ROUTES } from "@/lib/routes";
import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

export function CredentialsList() {
	const { t } = useTranslation();

	const {
		data: credentials,
		isLoading,
		error,
	} = useQuery({
		queryKey: ["credentials"],
		queryFn: getCredentials,
	});

	if (isLoading) {
		return <CredentialsSkeleton />;
	}

	if (error) {
		return <CredentialsError />;
	}

	return (
		<div className="h-full overflow-y-auto p-8 space-y-4">
			<div className="flex justify-end items-center">
				<Button asChild>
					<Link to={APP_ROUTES["settings-credentials-new-select"].to}>
						{t("common.create")}
					</Link>
				</Button>
			</div>
			{!credentials || credentials.length === 0 ? (
				<p className="text-sm text-muted-foreground">
					{t("components.settings.credentials.noCredentials")}
				</p>
			) : (
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
					{credentials.map((cred) => (
						<Credential key={cred.id} credential={cred} />
					))}
				</div>
			)}
		</div>
	);
}

function CredentialsSkeleton() {
	return (
		<div className="flex flex-col h-full">
			<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
				{Array.from({ length: 4 }).map((_, index) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
					<Skeleton key={index} className="h-32 w-full" />
				))}
			</div>
		</div>
	);
}

function CredentialsError() {
	const { t } = useTranslation();

	return (
		<div className="flex flex-col h-full">
			<div className="space-y-4 p-4">
				<p className="text-sm text-muted-foreground">
					{t("components.settings.credentials.error")}
				</p>
			</div>
		</div>
	);
}

function Credential({ credential }: { credential: UnifiedCredential }) {
	const { t } = useTranslation();
	const [isOpen, setIsOpen] = useState(false);

	const legacyData = useMemo(() => {
		if (credential.version === '1') {
			const displayName = credential.displayName;
			
			const serviceMatch = displayName.match(/^(\w+):\/\/(.+)@(.+)$/);
			if (serviceMatch) {
				const [, , user, host] = serviceMatch;
				return { user, host };
			}

			return { user: 'N/A', host: 'N/A' };
		}
		return null;
	}, [credential]);

	const ServiceIcon = getServiceIcon(credential.provider);

	const getStatusBadgeVariant = (status?: string) => {
		switch (status) {
			case 'crawling':
				return 'default';
			case 'idle':
				return 'secondary';
			case 'error_permanent':
				return 'destructive';
			default:
				return 'outline';
		}
	};

	const getStatusLabel = (status?: string) => {
		switch (status) {
			case 'crawling':
				return t("components.settings.credentials.status.crawling", "Crawling");
			case 'idle':
				return t("components.settings.credentials.status.idle", "Idle");
			case 'error_permanent':
				return t("components.settings.credentials.status.error", "Error");
			default:
				return t("components.settings.credentials.status.unknown", "Unknown");
		}
	};

	return (
		<Card>
			<CardHeader>
				<CardTitle className="flex justify-between items-center">
					<div className="flex items-center gap-3">
						<div className="w-6 h-6 flex items-center justify-center">
							<ServiceIcon className="w-5 h-5 text-primary" />
						</div>
						<div className="flex flex-col">
							<span>{getServiceLabel(credential.provider)}</span>
							{credential.status && (
								<Badge variant={getStatusBadgeVariant(credential.status)} className="text-xs w-fit">
									{getStatusLabel(credential.status)}
								</Badge>
							)}
						</div>
					</div>
					<DeleteCredentialButton credential={credential} />
				</CardTitle>
			</CardHeader>
			<CardContent className="text-sm space-y-1">
				<p className="break-all font-medium">
					{credential.displayName}
				</p>
				
				{credential.version === '1' && legacyData && (
					<>
						<p>
							<strong>{t("components.settings.credentials.host")}: </strong>
							{legacyData.host}
						</p>
						<p>
							<strong>{t("components.settings.credentials.username")}: </strong>
							{legacyData.user}
						</p>
						<p>
							<strong>{t("components.settings.credentials.password")}: </strong>
							********
						</p>
					</>
				)}

				{/* V2 Resources */}
				{credential.version === '2' && credential.resources && credential.resources.length > 0 && (
					<Collapsible open={isOpen} onOpenChange={setIsOpen}>
						<CollapsibleTrigger asChild>
							<Button variant="ghost" className="w-full justify-between h-auto p-2">
								<span className="text-sm font-medium">
									{t("components.settings.credentials.resources", "Resources")} ({credential.resources.length})
								</span>
								{isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
							</Button>
						</CollapsibleTrigger>
						<CollapsibleContent className="space-y-2 pt-2">
							{credential.resources.map((resource) => (
								<div key={resource.id} className="border rounded-md p-2 bg-muted/50">
									<div className="flex justify-between items-start">
										<div className="flex-1">
											<p className="font-medium text-sm">{resource.displayName}</p>
											<div className="flex items-center gap-2 mt-1">
												<Badge 
													variant={getStatusBadgeVariant(resource.status)} 
													className="text-xs"
												>
													{getStatusLabel(resource.status)}
												</Badge>
												<div className="flex items-center gap-1">
													{resource.isEnabled ? (
														<Eye className="h-3 w-3 text-green-600" />
													) : (
														<EyeOff className="h-3 w-3 text-gray-400" />
													)}
													<span className="text-xs text-muted-foreground">
														{resource.isEnabled 
															? t("components.settings.credentials.enabled", "Enabled")
															: t("components.settings.credentials.disabled", "Disabled")
														}
													</span>
												</div>
											</div>
										</div>
									</div>
								</div>
							))}
						</CollapsibleContent>
					</Collapsible>
				)}
			</CardContent>
		</Card>
	);
}

function DeleteCredentialButton({
	credential,
}: { credential: UnifiedCredential }) {
	const { t } = useTranslation();

	const [open, setOpen] = useState(false);

	const queryClient = useQueryClient();

	const { mutate: deleteCredentialMutation, isPending } = useMutation({
		mutationFn: deleteCredential,
		onSuccess: () => {
			toast.success(t("components.settings.credentials.delete.success"));
			queryClient.invalidateQueries({ queryKey: ["credentials"] });
			setOpen(false);
		},
		onError: (error: Error) => {
			toast.error(error.message || t("components.settings.credentials.delete.error"));
		},
	});

	return (
		<AlertDialog open={open} onOpenChange={setOpen}>
			<AlertDialogTrigger asChild>
				<Button variant="destructive" size="icon">
					<Trash2 className="h-4 w-4" />
					<span className="sr-only">{t("common.delete")}</span>
				</Button>
			</AlertDialogTrigger>
			<AlertDialogContent>
				<AlertDialogHeader>
					<AlertDialogTitle>
						{t("components.settings.credentials.delete.title")}
					</AlertDialogTitle>
					<AlertDialogDescription>
						{t("components.settings.credentials.delete.description")}
					</AlertDialogDescription>
				</AlertDialogHeader>
				<AlertDialogFooter>
					<AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
					<AlertDialogAction
						onClick={(e) => {
							e.preventDefault();
							deleteCredentialMutation(credential.id);
						}}
						disabled={isPending}
					>
						{isPending ? (
							<Loader2 className="w-4 h-4 animate-spin" />
						) : (
							t("common.delete")
						)}
					</AlertDialogAction>
				</AlertDialogFooter>
			</AlertDialogContent>
		</AlertDialog>
	);
}

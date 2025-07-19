import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useQuery } from "@tanstack/react-query";
import { fetchPathContents } from "@/services/path";
import { useMemo, useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
	FolderIcon,
	FileIcon,
	Loader2,
	ArrowLeftIcon,
	X,
	Plus,
	ChevronDownIcon,
	ChevronRightIcon,
	ServerIcon,
} from "lucide-react";
import type { PathContentItem, ServicePathContentsResult } from "@/types/path";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { getServiceIcon, getServiceLabel } from "@/lib/service-config";

interface FileExplorerPanelContentProps {
	selectedItems: PathContentItem[];
	onAddToContext: (item: PathContentItem) => void;
	onRemoveFromContext: (item: PathContentItem) => void;
}

interface FileExplorerPanelProps extends FileExplorerPanelContentProps {
	open: boolean;
	onClose: () => void;
}

export function FileExplorerPanel({
	open,
	onClose,
	selectedItems,
	onAddToContext,
	onRemoveFromContext,
}: FileExplorerPanelProps) {
	const { t } = useTranslation();
	if (!open) return null;
	return (
		<div className="flex-1 max-w-2xl h-full border-l bg-background flex flex-col z-10">
			<div className="flex items-center justify-between px-4 py-2">
				<span className="font-semibold text-base">
					{t("components.fileExplorer.title")}
				</span>
				<button
					className="rounded p-1 hover:bg-muted"
					onClick={onClose}
					title={t("common.close")}
					type="button"
				>
					<X className="h-5 w-5" />
				</button>
			</div>
			<FileExplorerPanelContent
				selectedItems={selectedItems}
				onAddToContext={onAddToContext}
				onRemoveFromContext={onRemoveFromContext}
			/>
		</div>
	);
}

export function FileExplorerPanelContent({
	selectedItems,
	onAddToContext,
	onRemoveFromContext,
}: FileExplorerPanelContentProps) {
	const { t } = useTranslation();
	const [currentPath, setCurrentPath] = useState("/");
	const [expandedServices, setExpandedServices] = useState<Set<number>>(new Set());

	const {
		data: pathResults,
		isLoading,
		error,
	} = useQuery({
		queryKey: ["pathContents", currentPath],
		queryFn: () => fetchPathContents(currentPath),
	});

	const handleFolderClick = (folderPath: string) => {
		setCurrentPath(folderPath);
	};

	const handleBackClick = () => {
		if (currentPath === "/") return;
		const parts = currentPath.split("/").filter(Boolean);
		parts.pop();
		setCurrentPath(`/${parts.join("/")}${parts.length > 0 ? "/" : ""}`);
	};

	const serviceGroups = useMemo(() => {
		if (!pathResults) return [];
		
		return pathResults
			.filter((result) => result.items && result.items.length > 0)
			.map((result) => ({
				...result,
				sortedItems: result.items.sort((a, b) => {
					if (a.type === "folder" && b.type !== "folder") return -1;
					if (a.type !== "folder" && b.type === "folder") return 1;
					return a.name.localeCompare(b.name);
				}),
			}));
	}, [pathResults]);

	const toggleServiceExpansion = (serviceId: number) => {
		setExpandedServices((prev) => {
			const newSet = new Set(prev);
			if (newSet.has(serviceId)) {
				newSet.delete(serviceId);
			} else {
				newSet.add(serviceId);
			}
			return newSet;
		});
	};

	// Auto-expand services when data loads
	useEffect(() => {
		if (serviceGroups.length > 0) {
			setExpandedServices(new Set(serviceGroups.map(group => group.serviceId)));
		}
	}, [serviceGroups]);

	const isSelected = (item: PathContentItem) =>
		selectedItems.some(
			(f) => f.path === item.path && f.serviceId === item.serviceId,
		);

	return (
		<div className="flex flex-col h-full w-full p-4">
			<div className="flex flex-col gap-1 mb-3">
				<div className="flex items-center text-sm">
					{currentPath !== "/" && (
						<Button
							variant="ghost"
							size="icon"
							onClick={handleBackClick}
							className="mr-2"
						>
							<ArrowLeftIcon className="h-4 w-4" />
						</Button>
					)}
					<span className="text-muted-foreground font-medium mr-2">
						{t("components.fileExplorer.currentPathLabel")}:
					</span>
					<span
						className="font-mono text-xs text-muted-foreground truncate"
						title={currentPath}
					>
						{currentPath}
					</span>
				</div>
				<div className="my-2">
					<div className="border-b border-border/60 w-full" />
				</div>
			</div>
			<div className="flex-1 min-h-0">
				<ScrollArea className="h-full w-full rounded-md">
					{isLoading && (
						<div className="flex justify-center items-center h-full">
							<Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
						</div>
					)}
					{error && (
						<Alert variant="destructive">
							<AlertTitle>{t("common.error")}</AlertTitle>
							<AlertDescription>{error.message}</AlertDescription>
						</Alert>
					)}
					{!isLoading && !error && serviceGroups.length === 0 && (
						<div className="flex justify-center items-center h-full">
							<p className="text-muted-foreground">
								{t("components.fileExplorer.empty")}
							</p>
						</div>
					)}
					{!isLoading && !error && serviceGroups.length > 0 && (
						<div className="space-y-3">
							{serviceGroups.map((serviceGroup) => {
								const isExpanded = expandedServices.has(serviceGroup.serviceId);
								const serviceName = serviceGroup.serviceMetadata?.serviceName || 
									serviceGroup.serviceMetadata?.serviceProvider || 
									`Service ${serviceGroup.serviceId}`;
								const serviceHost = serviceGroup.serviceMetadata?.host;
								
								// Get the service icon and label based on the service provider
								const ServiceIcon = getServiceIcon(
									serviceGroup.serviceMetadata?.serviceProvider || 
									serviceGroup.serviceMetadata?.serviceName || 
									'unknown'
								);
								
								const serviceLabel = getServiceLabel(
									serviceGroup.serviceMetadata?.serviceProvider || 
									serviceGroup.serviceMetadata?.serviceName || 
									'unknown'
								);
								
								// Create a more user-friendly display name
								const serviceDisplayName = serviceHost 
									? `${serviceLabel} (${serviceHost})`
									: serviceLabel;
								
								return (
									<Collapsible 
										key={serviceGroup.serviceId}
										open={isExpanded}
										onOpenChange={() => toggleServiceExpansion(serviceGroup.serviceId)}
									>
										<CollapsibleTrigger asChild>
											<Button
												variant="ghost"
												className="w-full justify-start py-2 px-2 h-auto font-medium text-sm"
											>
												{isExpanded ? (
													<ChevronDownIcon className="h-4 w-4 mr-2" />
												) : (
													<ChevronRightIcon className="h-4 w-4 mr-2" />
												)}
												<div className="w-4 h-4 mr-2 flex items-center justify-center">
													<ServiceIcon className="h-4 w-4 text-primary" />
												</div>
												<span className="truncate" title={serviceDisplayName}>
													{serviceDisplayName}
												</span>
												<span className="ml-auto text-xs text-muted-foreground">
													({serviceGroup.sortedItems.length} items)
												</span>
											</Button>
										</CollapsibleTrigger>
										<CollapsibleContent className="pl-4 pt-1">
											<ul className="space-y-0.5">
												{serviceGroup.sortedItems.map((item) => (
													<li
														key={`${item.serviceId}-${item.path}`}
														className="flex items-center"
													>
														<Button
															variant="ghost"
															className="w-full justify-start h-auto py-1 px-1.5 text-sm flex items-center"
															onClick={() =>
																item.type === "folder"
																	? handleFolderClick(item.path)
																	: onAddToContext(item)
															}
														>
															{item.type === "folder" ? (
																<FolderIcon className="h-3 w-3 mr-2 text-primary flex-shrink-0" />
															) : (
																<FileIcon className="h-3 w-3 mr-2 text-muted-foreground flex-shrink-0" />
															)}
															<span
																className="truncate flex-grow min-w-0 text-start"
																title={item.name}
																style={{ minWidth: 0 }}
															>
																{item.name}
															</span>
														</Button>
														<Button
															variant="ghost"
															size="icon"
															className="ml-1 h-6 w-6 flex-shrink-0"
															onClick={() => onAddToContext(item)}
															disabled={isSelected(item)}
															tabIndex={-1}
															title={item.name || "Add to context"}
														>
															<span className="sr-only">Add to context</span>
															<Plus className="h-4 w-4" />
														</Button>
													</li>
												))}
											</ul>
										</CollapsibleContent>
									</Collapsible>
								);
							})}
						</div>
					)}
				</ScrollArea>
			</div>
		</div>
	);
}

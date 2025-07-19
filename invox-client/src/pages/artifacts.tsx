import { useState, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router-dom";
import { useInfiniteQuery } from "@tanstack/react-query";
import { Loader2, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { getArtifacts } from "@/services/artifacts";
import { previewCache } from "@/services/preview-cache";
import { ArtifactsHeader } from "@/components/artifacts/v1/artifacts-header";
import { ArtifactListItem } from "@/components/artifacts/v1/artifact-list-item";
import { ArtifactsSkeleton } from "@/components/artifacts/v1/artifacts-skeleton";
import { FILE_TYPE_CATEGORIES } from "@/components/artifacts/v1/artifact-filters";
import type { ArtifactFilters, FileCategories } from "@/types/artifacts";

const DOCUMENTS_PER_PAGE = 20;

function parseFiltersFromURL(searchParams: URLSearchParams): ArtifactFilters {
	const filters: ArtifactFilters = {};
	
	const fileTypes = searchParams.getAll("fileType");
	if (fileTypes.length > 0) {
		filters.fileTypes = fileTypes as FileCategories[];
	}
	
	return filters;
}

function updateURLWithFilters(searchParams: URLSearchParams, filters: ArtifactFilters, searchQuery: string) {
	const newParams = new URLSearchParams();
	
	if (searchQuery) {
		newParams.set("q", searchQuery);
	}
	
	filters.fileTypes?.forEach(type => newParams.append("fileType", type));
	
	return newParams;
}

export function ArtifactsPage() {
	const { t } = useTranslation();
	const [searchParams, setSearchParams] = useSearchParams();
	
	const [searchQuery, setSearchQuery] = useState(searchParams.get("q") || "");
	const [filters, setFilters] = useState<ArtifactFilters>(() => parseFiltersFromURL(searchParams));

	const activeMimeTypes = filters.fileTypes?.flatMap(category => FILE_TYPE_CATEGORIES[category] || []) || [];
	const hasClientSideFiltering = activeMimeTypes.length > 0;

	const {
		data,
		fetchNextPage,
		hasNextPage,
		isFetchingNextPage,
		isLoading,
		isError,
		error,
		refetch,
	} = useInfiniteQuery({
		queryKey: ["documents-v1", searchQuery, filters],
		queryFn: async ({ pageParam = 0 }) => {
			const result = await getArtifacts({
				text: searchQuery || undefined,
				offset: (pageParam as number) * DOCUMENTS_PER_PAGE,
				size: DOCUMENTS_PER_PAGE,
				filters,
			});
			
			return result;
		},
		getNextPageParam: (lastPage, allPages) => {
			if (hasClientSideFiltering) {
				return undefined;
			}
			return lastPage.metadata.artifacts.hasMore ? allPages.length : undefined;
		},
		initialPageParam: 0,
		staleTime: 5 * 60 * 1000, // 5 minutes
	});

	const handleSearch = useCallback((query: string) => {
		setSearchQuery(query);
		// Update URL immediately
		const newParams = updateURLWithFilters(searchParams, filters, query);
		setSearchParams(newParams, { replace: true });
	}, [filters, searchParams, setSearchParams]);

	const handleFiltersChange = useCallback((newFilters: ArtifactFilters) => {
		setFilters(newFilters);
		// Update URL immediately
		const newParams = updateURLWithFilters(searchParams, newFilters, searchQuery);
		setSearchParams(newParams, { replace: true });
	}, [searchQuery, searchParams, setSearchParams]);

	const handleLoadMore = () => {
		if (hasNextPage && !isFetchingNextPage) {
			fetchNextPage();
		}
	};

	const allDocuments = data?.pages?.flatMap((page) => page.artifacts) || [];
	const totalCount = data?.pages?.[0]?.metadata.artifacts.totalCount || 0;

	useEffect(() => {
		if (allDocuments.length > 0) {
			const documentIds = allDocuments.map(doc => doc.id);
			previewCache.prefetchPreviews(documentIds).catch(error => {
				console.warn("Failed to prefetch some preview images:", error);
			});
		}
	}, [allDocuments]);

	useEffect(() => {
		const urlQuery = searchParams.get("q") || "";
		const urlFilters = parseFiltersFromURL(searchParams);
		
		const currentFiltersStr = JSON.stringify(filters);
		const urlFiltersStr = JSON.stringify(urlFilters);
		
		if (urlQuery !== searchQuery) {
			setSearchQuery(urlQuery);
		}
		
		if (urlFiltersStr !== currentFiltersStr) {
			setFilters(urlFilters);
		}
	}, [searchParams]);

	if (isLoading) {
		return <ArtifactsSkeleton viewMode="list" />;
	}

	if (isError) {
		return (
			<div className="flex flex-col h-full">
				<ArtifactsHeader
					searchQuery={searchQuery}
					onSearch={handleSearch}
					filters={filters}
					onFiltersChange={handleFiltersChange}
				/>
				<div className="flex-1 flex items-center justify-center p-6">
					<Alert className="max-w-md">
						<AlertDescription>
							{t("components.artifacts.error")}: {error?.message || t("common.error")}
							<Button
								variant="outline"
								size="sm"
								onClick={() => refetch()}
								className="mt-2"
							>
								{t("common.retry")}
							</Button>
						</AlertDescription>
					</Alert>
				</div>
			</div>
		);
	}

	return (
		<div className="flex flex-col h-full">
			<ArtifactsHeader
				searchQuery={searchQuery}
				onSearch={handleSearch}
				filters={filters}
				onFiltersChange={handleFiltersChange}
				totalCount={totalCount}
			/>

			<div className="flex-1 overflow-hidden">
				{allDocuments.length === 0 ? (
					<div className="flex flex-col items-center justify-center h-full p-6">
						<FileText className="w-16 h-16 text-muted-foreground mb-4" />
						<h3 className="text-lg font-semibold mb-2">
							{searchQuery ? t("components.artifacts.noResults") : t("components.artifacts.noArtifacts")}
						</h3>
						<p className="text-muted-foreground text-center">
							{searchQuery 
								? t("components.artifacts.noResultsDescription")
								: t("components.artifacts.noArtifactsDescription")
							}
						</p>
					</div>
				) : (
					<ScrollArea className="h-full">
						<div className="p-6">
							<div className="space-y-4 max-w-screen-xl mx-auto">
								{allDocuments.map((artifact) => (
									<ArtifactListItem
										key={artifact.id}
										artifact={artifact}
									/>
								))}
							</div>

							{hasNextPage && !hasClientSideFiltering && (
								<div className="flex justify-center py-8">
									<Button
										variant="outline"
										onClick={handleLoadMore}
										disabled={isFetchingNextPage}
										className="flex items-center gap-2"
									>
										{isFetchingNextPage ? (
											<>
												<Loader2 className="w-4 h-4 animate-spin" />
												{t("components.artifacts.loadingMore")}
											</>
										) : (
											t("components.artifacts.loadMore")
										)}
									</Button>
								</div>
							)}

							{hasClientSideFiltering && allDocuments.length > 0 && (
								<div className="text-center py-8">
									<p className="text-sm text-muted-foreground">
										{t("components.artifacts.clientSideFiltering")}
									</p>
								</div>
							)}

							{!hasNextPage && allDocuments.length > 0 && !hasClientSideFiltering && (
								<div className="text-center py-8">
									<p className="text-sm text-muted-foreground">
										{t("components.artifacts.allLoaded")}
									</p>
								</div>
							)}
						</div>
					</ScrollArea>
				)}
			</div>
		</div>
	);
} 
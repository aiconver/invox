import { ArtifactSearch } from "./artifact-search";
import { ArtifactFilters } from "./artifact-filters";
import { useTranslation } from "react-i18next";
import type { ArtifactFilters as ArtifactFiltersType } from "@/types/artifacts";

interface ArtifactsHeaderProps {
	searchQuery: string;
	onSearch: (query: string) => void;
	filters: ArtifactFiltersType;
	onFiltersChange: (filters: ArtifactFiltersType) => void;
	totalCount?: number;
}

export function ArtifactsHeader({ 
	searchQuery, 
	onSearch,
	filters,
	onFiltersChange,
	totalCount,
}: ArtifactsHeaderProps) {
	const { t } = useTranslation();

	return (
		<div className="border-b">
			<div className="p-4">
				<div className="flex items-center justify-between gap-4">
					<div className="flex-1">
						<ArtifactSearch onSearch={onSearch} initialValue={searchQuery} />
					</div>
				</div>
				
				{totalCount !== undefined && totalCount > 0 && (
					<p className="ml-2 text-sm text-muted-foreground mt-2">
						{t("components.artifacts.resultsCount", { count: totalCount })}
					</p>
				)}
			</div>
			
			<ArtifactFilters 
				filters={filters}
				onFiltersChange={onFiltersChange}
			/>
		</div>
	);
} 
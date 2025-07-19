import { useTranslation } from "react-i18next";
import { Filter, X, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuCheckboxItem,
	DropdownMenuTrigger,
	DropdownMenuSeparator,
	DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import type { ArtifactFilters, FileCategories } from "@/types/artifacts";

interface ArtifactFiltersProps {
	filters: ArtifactFilters;
	onFiltersChange: (filters: ArtifactFilters) => void;
}

const FILE_TYPE_CATEGORIES: Record<FileCategories, string[]> = {
	Documents: [
		"application/pdf",
		"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		"application/msword",
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		"application/vnd.ms-excel",
		"text/csv",
		"text/plain",
		"text/html",
		"application/rtf",
	],
	Presentations: [
		"application/vnd.openxmlformats-officedocument.presentationml.presentation",
		"application/vnd.ms-powerpoint",
	],
	Images: ["image/jpeg", "image/png", "image/gif", "image/svg+xml", "image/webp"],
	Videos: ["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-ms-wmv", "video/webm", "video/x-matroska"],
	Audio: ["audio/mpeg", "audio/wav", "audio/ogg", "audio/webm", "audio/x-aac", "audio/flac"],
	Archives: ["application/zip", "application/vnd.rar", "application/x-7z-compressed"],
	Data: ["application/json", "application/xml", "text/xml"],
};

export function ArtifactFilters({ filters, onFiltersChange }: ArtifactFiltersProps) {
	const { t } = useTranslation();

	const hasActiveFilters = filters.fileTypes && filters.fileTypes.length > 0;

	const handleFileTypeToggle = (category: FileCategories) => {
		const currentTypes = filters.fileTypes || [];
		const newTypes = currentTypes.includes(category)
			? currentTypes.filter(t => t !== category)
			: [...currentTypes, category];
		
		onFiltersChange({
			...filters,
			fileTypes: newTypes.length > 0 ? newTypes : undefined,
		});
	};

	const clearAllFilters = () => {
		onFiltersChange({});
	};

	const removeFileTypeFilter = (category: string) => {
		const newTypes = (filters.fileTypes || []).filter(t => t !== category);
		onFiltersChange({
			...filters,
			fileTypes: newTypes.length > 0 ? newTypes : undefined,
		});
	};

	return (
		<div className="flex flex-col gap-3 p-4 border-b bg-muted/30">
			<div className="flex items-center gap-2 flex-wrap">
				<DropdownMenu>
					<DropdownMenuTrigger asChild>
						<Button 
							variant={hasActiveFilters ? "default" : "outline"} 
							size="sm"
							className="flex items-center gap-1"
						>
							<Filter className="w-3 h-3" />
							{t("components.artifacts.filters.fileType")}
							{filters.fileTypes?.length ? ` (${filters.fileTypes.length})` : ""}
							<ChevronDown className="w-3 h-3" />
						</Button>
					</DropdownMenuTrigger>
					<DropdownMenuContent className="w-56">
						<DropdownMenuLabel>{t("components.artifacts.filters.fileType")}</DropdownMenuLabel>
						<DropdownMenuSeparator />
						{Object.keys(FILE_TYPE_CATEGORIES).map((category) => (
							<DropdownMenuCheckboxItem
								key={category}
								checked={filters.fileTypes?.includes(category as FileCategories) || false}
								onCheckedChange={() => handleFileTypeToggle(category as FileCategories)}
							>
								{t(`components.artifacts.filters.categories.${category}`, category)}
							</DropdownMenuCheckboxItem>
						))}
					</DropdownMenuContent>
				</DropdownMenu>

				{hasActiveFilters && (
					<Button 
						variant="ghost" 
						size="sm"
						onClick={clearAllFilters}
						className="flex items-center gap-1 text-muted-foreground hover:text-foreground"
					>
						<X className="w-3 h-3" />
						{t("components.artifacts.filters.clearAll")}
					</Button>
				)}
			</div>

			{hasActiveFilters && (
				<div className="flex items-center gap-2 flex-wrap">
					{filters.fileTypes?.map((category) => (
						<Badge key={`type-${category}`} variant="secondary" className="flex items-center gap-1">
							{t(`components.artifacts.filters.categories.${category}`, category)}
							<Button
								variant="ghost"
								size="sm"
								onClick={() => removeFileTypeFilter(category)}
								className="h-3 w-3 p-0 hover:bg-transparent"
							>
								<X className="w-2 h-2" />
							</Button>
						</Badge>
					))}
				</div>
			)}
		</div>
	);
}

export { FILE_TYPE_CATEGORIES }; 
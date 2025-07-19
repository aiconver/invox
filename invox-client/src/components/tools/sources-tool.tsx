import type { ToolSource } from "@/types/tools";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Link2 } from "lucide-react";
import { FetchPdfButton } from "../fetch-pdf-button";
import { useTranslation } from "react-i18next";

export interface ToolInvocationSourceProps {
	source: ToolSource;
}

export function ToolInvocationSource({ source }: ToolInvocationSourceProps) {
	const { t } = useTranslation();
	return (
		<FetchPdfButton
			pdfId={source.documentId}
			pdfPage={source.pageNumber.toString()}
			asChild
		>
			<div className="group/source flex items-start gap-2 py-1 hover:bg-muted/50 px-2 rounded-md transition-colors w-full overflow-hidden">
				<Link2 className="h-4 w-4 mt-1 shrink-0 text-muted-foreground group-hover/source:text-foreground" />

				<div className="w-full flex-1 flex flex-col items-start gap-1 overflow-hidden min-w-0">
					<div className="text-sm font-medium line-clamp-1 no-underline group-hover/source:underline">
						{source.documentName}
					</div>
					<div className="text-sm font-medium line-clamp-1 no-underline group-hover/source:underline">
						{t("common.page")}: {source.pageNumber}
					</div>

					<div
						className={cn(
							"text-xs text-muted-foreground line-clamp-1 break-all",
							"group-hover/source:underline",
						)}
					>
						{source.documentId}
					</div>
				</div>
			</div>
		</FetchPdfButton>
	);
}

export interface ToolInvocationSourcesListProps {
	sources: ToolSource[];
	maxVisible?: number;
	maxHeight?: string;
}

export function ToolInvocationSourcesList({
	sources,
}: ToolInvocationSourcesListProps) {
	const { t } = useTranslation();
	return (
		<div className="w-full overflow-y-auto flex flex-col gap-3 py-1">
			<div className="text-sm font-medium ml-2">
				{t("common.sources")} ({sources.length})
			</div>
			<ScrollArea className="h-full max-h-[10rem]">
				<div className="flex flex-col gap-1">
					{sources.map((source, index) => (
						<ToolInvocationSource
							key={`source-${source.documentId}-${index}`}
							source={source}
						/>
					))}
				</div>
			</ScrollArea>
		</div>
	);
}

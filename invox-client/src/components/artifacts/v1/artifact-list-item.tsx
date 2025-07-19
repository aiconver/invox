import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { 
	Download, 
	MessageSquare,
	Calendar,
	FileText,
	Users,
	Clock,
	Hash,
	Info,
	Database,
	MoreHorizontal
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Artifact } from "@/types/artifacts";
import { ArtifactPreview } from "./artifact-preview";
import { getLanguageName } from "@/lib/language-flags";
import { getServiceIcon, getServiceLabel } from "@/lib/service-config";
import { APP_ROUTES } from "@/lib/routes";
import { useFetchPdf } from "@/hooks/use-fetch-pdf";
import { FetchPdfButton } from "@/components/fetch-pdf-button";

type ViewMode = "summary" | "metadata";

interface ArtifactListItemProps {
	artifact: Artifact;
}

export function ArtifactListItem({ artifact }: ArtifactListItemProps) {
	const { t } = useTranslation();
	const navigate = useNavigate();
	const [viewMode, setViewMode] = useState<ViewMode>("summary");
	const { isFetching: isDownloading, fetchPdf } = useFetchPdf();

	const handleQANavigation = (e: React.MouseEvent) => {
		e.stopPropagation();
		// Navigate to QA page with this document as context
		navigate(`${APP_ROUTES["questions-answers"].to}?document=${artifact.id}`);
	};

	const formatDate = (dateString: string) => {
		try {
			return new Date(dateString).toLocaleDateString(undefined, {
				year: 'numeric',
				month: 'short',
				day: 'numeric'
			});
		} catch {
			return dateString;
		}
	};

	const getDisplaySummary = () => {
		if (artifact.summaryShortGenerated) return artifact.summaryShortGenerated;
		if (artifact.summaryShort) return artifact.summaryShort;
		if (artifact.summaryLongGenerated) return artifact.summaryLongGenerated;
		if (artifact.summaryLong) return artifact.summaryLong;
		return null;
	};

	const getLongSummary = () => {
		if (artifact.summaryLongGenerated) return artifact.summaryLongGenerated;
		if (artifact.summaryLong) return artifact.summaryLong;
		return null;
	};

	const shortSummary = getDisplaySummary();
	const longSummary = getLongSummary();

	return (
		<Card className="w-full hover:shadow-lg transition-all duration-200 group">
			<CardContent className="p-4">
				{/* Mobile: Stack vertically, Desktop: Side by side */}
				<div className="flex flex-col lg:flex-row gap-6 lg:h-80">
					{/* Preview Section - Increased width on desktop */}
					<div className="flex-shrink-0 relative w-full lg:w-96">
						<ArtifactPreview 
							documentId={artifact.id}
							documentName={artifact.name}
							language={artifact.language}
							artifact={artifact}
							className="w-full h-64 lg:h-full group-hover:shadow-md transition-shadow duration-200"
						/>
					</div>

					{/* Content Section */}
					<div className="flex-1 min-w-0 flex flex-col">
						{/* Title - Always visible, non-scrolling */}
						<div className="pb-4 border-b mb-4 flex-shrink-0">
							<h3 className="font-semibold text-xl line-clamp-2 leading-tight">
								{artifact.name}
							</h3>
						</div>

						{/* Scrollable content */}
						<div className="flex-1 overflow-hidden">
							<ScrollArea className="h-full">
								<div className="space-y-4 pr-2">
									{viewMode === "summary" && (
										<SummaryView 
											artifact={artifact} 
											shortSummary={shortSummary}
											longSummary={longSummary}
										/>
									)}

									{viewMode === "metadata" && (
										<MetadataView 
											artifact={artifact} 
											formatDate={formatDate}
										/>
									)}
								</div>
							</ScrollArea>
						</div>

						{/* Action Buttons */}
						<div className="mt-4 pt-4 border-t flex-shrink-0">
							<div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
								<div className="flex gap-2 order-2 md:order-1">
									<Button
										variant={viewMode === "summary" ? "default" : "outline"}
										size="sm"
										onClick={() => setViewMode("summary")}
										className="flex items-center gap-1"
									>
										<FileText className="w-4 h-4" />
										{t("common.overview")}
									</Button>
									<Button
										variant={viewMode === "metadata" ? "default" : "outline"}
										size="sm"
										onClick={() => setViewMode("metadata")}
										className="flex items-center gap-1"
									>
										<Info className="w-4 h-4" />
										{t("common.details")}
									</Button>
								</div>

								<div className="flex gap-2 order-1 md:order-2">
									<Button
										variant="outline"
										size="sm"
										onClick={handleQANavigation}
										className="flex items-center gap-1"
										title={t("common.askQuestions")}
									>
										<MessageSquare className="w-4 h-4" />
										{t("common.qa")}
									</Button>
									<FetchPdfButton
										pdfId={artifact.id}
										filename={`${artifact.name}.pdf`}
										download
										asChild
									>
										<Button
											variant="outline"
											size="sm"
											disabled={isDownloading}
											className="flex items-center gap-1"
										>
											<Download className="w-4 h-4" />
											{isDownloading ? t("common.downloading") : t("common.download")}
										</Button>
									</FetchPdfButton>
								</div>
							</div>
						</div>
					</div>
				</div>
			</CardContent>
		</Card>
	);
}

function SummaryView({ 
	artifact, 
	shortSummary, 
	longSummary
}: { 
	artifact: Artifact; 
	shortSummary: string | null;
	longSummary: string | null;
}) {
	const { t } = useTranslation();

	return (
		<div className="space-y-4">
			{/* Short Summary - No header, just content */}
			{shortSummary && (
				<div>
					<p className="text-sm leading-relaxed text-gray-700">
						{shortSummary}
					</p>
				</div>
			)}

			{/* Detailed Summary */}
			{longSummary && longSummary !== shortSummary && (
				<div>
					<h4 className="font-medium text-sm mb-2">
						{t("components.artifacts.detailedSummary")}
					</h4>
					<p className="text-sm leading-relaxed text-gray-600">
						{longSummary}
					</p>
				</div>
			)}
		</div>
	);
}

function MetadataView({ 
	artifact, 
	formatDate 
}: { 
	artifact: Artifact; 
	formatDate: (date: string) => string;
}) {
	const { t } = useTranslation();

	return (
		<div className="space-y-6">
			{/* Document Information */}
			<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div className="space-y-3">
					{artifact.pageCount && (
						<div className="flex items-center gap-2 text-sm">
							<FileText className="w-4 h-4 text-muted-foreground" />
							<span className="font-medium">{t("common.pages")}:</span>
							<span>{artifact.pageCount}</span>
						</div>
					)}
					<div className="flex items-center gap-2 text-sm">
						<Calendar className="w-4 h-4 text-muted-foreground" />
						<span className="font-medium">{t("common.modified")}:</span>
						<span>{formatDate(artifact.modification)}</span>
					</div>
					{artifact.readingTimeMinutes && (
						<div className="flex items-center gap-2 text-sm">
							<Clock className="w-4 h-4 text-muted-foreground" />
							<span className="font-medium">{t("common.readingTime")}:</span>
							<span>{artifact.readingTimeMinutes} {t("common.minutes")}</span>
						</div>
					)}
					{artifact.tokenCount && (
						<div className="flex items-center gap-2 text-sm">
							<Hash className="w-4 h-4 text-muted-foreground" />
							<span className="font-medium">{t("common.tokens")}:</span>
							<span>{artifact.tokenCount.toLocaleString()}</span>
						</div>
					)}
				</div>

				<div className="space-y-3">
					{artifact.language && (
						<div className="flex items-center gap-2 text-sm">
							<span className="font-medium">{t("common.language")}:</span>
							<Badge variant="secondary">
								{getLanguageName(artifact.language)}
							</Badge>
						</div>
					)}
					{artifact.type && (
						<div className="flex items-center gap-2 text-sm">
							<span className="font-medium">{t("common.type")}:</span>
							<Badge variant="outline">
								{artifact.type}
							</Badge>
						</div>
					)}
					{artifact.serviceName && (
						<div className="flex items-center gap-2 text-sm">
							<span className="font-medium">{t("common.service")}:</span>
							<Badge variant="outline" className="flex items-center gap-1">
								{(() => {
									const ServiceIcon = getServiceIcon(artifact.serviceName);
									return <ServiceIcon className="w-3 h-3" />;
								})()}
								{getServiceLabel(artifact.serviceName)}
							</Badge>
						</div>
					)}
				</div>
			</div>

			{/* Authors */}
			{artifact.authors && artifact.authors.length > 0 && (
				<div>
					<div className="flex items-center gap-2 mb-3">
						<Users className="w-4 h-4 text-muted-foreground" />
						<span className="text-sm font-medium">{t("common.authors")}</span>
					</div>
					<div className="flex flex-wrap gap-1">
						{artifact.authors.map((author, index) => (
							<Badge key={index} variant="outline" className="text-xs">
								{author.name}
							</Badge>
						))}
					</div>
				</div>
			)}

			{/* Owners */}
			{artifact.owners && artifact.owners.length > 0 && (
				<div>
					<div className="flex items-center gap-2 mb-3">
						<Users className="w-4 h-4 text-muted-foreground" />
						<span className="text-sm font-medium">{t("common.owners")}</span>
					</div>
					<div className="flex flex-wrap gap-1">
						{artifact.owners.map((owner, index) => (
							<Badge key={index} variant="outline" className="text-xs">
								{owner.name}
							</Badge>
						))}
					</div>
				</div>
			)}

			{/* Technical Details */}
			<div>
				<div className="flex items-center gap-2 mb-3">
					<Database className="w-4 h-4 text-muted-foreground" />
					<span className="text-sm font-medium">{t("components.artifacts.technicalDetails")}</span>
				</div>
				<div className="space-y-3">
					<div className="flex items-center gap-2 text-sm">
						<span className="font-medium">ID:</span> 
						<span className="font-mono text-gray-600 break-all">{artifact.id}</span>
					</div>
					<div className="flex items-center gap-2 text-sm">
						<span className="font-medium">Original ID:</span> 
						<span className="font-mono text-gray-600 break-all">{artifact.originalId}</span>
					</div>
					<div className="flex items-center gap-2 text-sm">
						<span className="font-medium">Service ID:</span> 
						<span className="font-mono text-gray-600">{artifact.serviceId}</span>
					</div>
					<div className="flex items-center gap-2 text-sm">
						<span className="font-medium">Parent:</span> 
						<span className="font-mono text-gray-600">{artifact.parent || 'None'}</span>
					</div>
					<div>
						<div className="flex items-center gap-2 text-sm mb-2">
							<span className="font-medium">Original Path:</span>
						</div>
						<div className="font-mono text-xs break-all text-gray-600 leading-relaxed">
							{artifact.originalPath}
						</div>
					</div>
				</div>
			</div>
		</div>
	);
} 
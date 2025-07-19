import { useState, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { Clock, FilesIcon, FileText, FileType2, FileQuestion } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { previewCache } from "@/services/preview-cache";
import { getLanguageFlagPath, getLanguageName } from "@/lib/language-flags";
import type { Artifact } from "@/types/artifacts";

interface ArtifactPreviewProps {
	documentId: string;
	documentName: string;
	language: string;
	artifact?: Artifact; // For showing additional quick info
	className?: string;
}

const getFileTypeIcon = (type: string | undefined): React.ElementType => {
    if (!type) return FileQuestion;
    const lowerType = type.toLowerCase();
    if (lowerType.includes("pdf")) return FileType2;
    return FileText;
};


export function ArtifactPreview({ 
	documentId, 
	documentName, 
	className = "w-32 h-40",
	language,
	artifact,
}: ArtifactPreviewProps) {
	const { t } = useTranslation();
	const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null);
	const [isLoadingPreview, setIsLoadingPreview] = useState(false);
	const [previewError, setPreviewError] = useState(false);
	const containerRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const loadPreviewImage = async () => {
			try {
				setIsLoadingPreview(true);
				setPreviewError(false);
				
				const cachedUrl = previewCache.getCachedUrl(documentId);
				if (cachedUrl) {
					setPreviewImageUrl(cachedUrl);
					setIsLoadingPreview(false);
					return;
				}

				const url = await previewCache.getPreviewUrl(documentId);
				setPreviewImageUrl(url);
			} catch (error) {
				console.error("Failed to load preview image:", error);
				setPreviewError(true);
			} finally {
				setIsLoadingPreview(false);
			}
		};

		loadPreviewImage();
	}, [documentId]);

	const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
		if (!containerRef.current) { return; }
		
		const container = containerRef.current;
		const rect = container.getBoundingClientRect();
		const y = e.clientY - rect.top;
		const containerHeight = rect.height;
		
		const scrollMargin = 96;
		const scrollPercent = Math.max(0, Math.min(1, (y - scrollMargin) / (containerHeight - 2 * scrollMargin )));
		
		container.style.setProperty('--scroll-y', `${scrollPercent * -50}%`);
	};

	const handleMouseLeave = () => {
		if (!containerRef.current) { return; }
		containerRef.current.style.setProperty('--scroll-y', '0%');
	};

	const renderContent = () => {
		if (isLoadingPreview) {
			return <FileText className="w-8 h-8 text-gray-400" />;
		}
		if (previewImageUrl && !previewError) {
			return (
				<div className="w-full h-full overflow-hidden p-2">
					<img
						src={previewImageUrl}
						alt={t("components.artifacts.previewImage", { name: documentName })}
						className="w-full h-[150%] object-contain object-top transition-transform duration-200 ease-out"
						style={{
							transform: 'translateY(var(--scroll-y))'
						}}
						onError={() => setPreviewError(true)}
					/>
				</div>
			);
		}
		return <FileText className="w-8 h-8 text-gray-400" />;
	};

	return (
		<div 
			ref={containerRef}
			className={`relative overflow-hidden rounded border cursor-pointer bg-gray-100 flex items-center justify-center ${className}`}
			onMouseMove={handleMouseMove}
			onMouseLeave={handleMouseLeave}
			style={{ '--scroll-y': '0%' } as React.CSSProperties}
		>
			{renderContent()}
			{artifact && (
				<ArtifactPreviewQuickInfo artifact={artifact} language={language} />
			)}
		</div>
	);
} 

function ArtifactPreviewQuickInfo({ artifact, language }: { artifact: Artifact, language: string }) {
	const { t } = useTranslation();
	const FileTypeIcon = getFileTypeIcon(artifact.type);

	return (
		<>
			{/* Bottom left info */}
			<div className="absolute bottom-2 left-2 flex flex-row gap-1.5 items-center">
				{artifact.type && (
					<Tooltip>
						<TooltipTrigger>
							<div className="bg-black/70 text-white rounded-md p-1 hover:bg-black/80 transition-colors">
								<FileTypeIcon className="w-4 h-4" />
							</div>
						</TooltipTrigger>
						<TooltipContent>
							<p>{t("common.type")}: {artifact.type}</p>
						</TooltipContent>
					</Tooltip>
				)}
				<Tooltip>
					<TooltipTrigger>
						<img
							src={getLanguageFlagPath(language)}
							alt={getLanguageName(language)}
							className="size-6 shrink-0 rounded-full shadow-sm border border-white/50"
						/>
					</TooltipTrigger>
					<TooltipContent>
						<p>{getLanguageName(language)}</p>
					</TooltipContent>
				</Tooltip>
			</div>
			
			{/* Bottom right badges */}
			<div className="absolute bottom-2 right-2 flex flex-row gap-1 items-end">
				{artifact.pageCount && (
					<Tooltip>
						<TooltipTrigger asChild>
							<Badge variant="secondary" className="text-xs bg-black/70 text-white hover:bg-black/80 px-2 py-1">
							<FilesIcon className="w-3.5 h-3.5 mr-1" />
								{artifact.pageCount}
							</Badge>
						</TooltipTrigger>
						<TooltipContent>
							<p>{artifact.pageCount} {t("common.pages")}</p>
						</TooltipContent>

					</Tooltip>
				)}
				{artifact.readingTimeMinutes && (
					<Tooltip>
						<TooltipTrigger asChild>
							<Badge variant="secondary" className="text-xs bg-black/70 text-white hover:bg-black/80 px-2 py-1">
							<Clock className="w-3.5 h-3.5 mr-1" />
								{artifact.readingTimeMinutes}
							</Badge>
						</TooltipTrigger>
						<TooltipContent>
							<p>{t("common.readingTime")}: {artifact.readingTimeMinutes} {t("common.minutes")}</p>
						</TooltipContent>
					</Tooltip>
				)}
			</div>
		</>
	);
}
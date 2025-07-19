import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
	ToolInvocation,
	ToolInvocationContent,
	ToolInvocationHeader,
	ToolInvocationName,
} from "@/components/ui/tool-invocation";
import { cn } from "@/lib/utils";
import type { ToolAnnotation } from "@/types/tools";

import { useState } from "react";
import { useTranslation } from "react-i18next";

interface AnnotatedToolProps {
	type: "call" | "result";
	annotations: ToolAnnotation[];
	titleCall: string;
	titleResult: string;
}

export function AnnotatedTool({
	type,
	titleCall,
	titleResult,
	annotations,
}: AnnotatedToolProps) {
	if (type === "call") {
		return (
			<AnnotatedToolCall titleCall={titleCall} annotations={annotations} />
		);
	}
	return (
		<AnnotatedToolResult titleResult={titleResult} annotations={annotations} />
	);
}

interface AnnotatedToolResultProps {
	titleResult: string;
	annotations: ToolAnnotation[];
}

function AnnotatedToolResult({
	titleResult,
	annotations,
}: AnnotatedToolResultProps) {
	const [historyOpen, setHistoryOpen] = useState(false);
	const sortedAnnotations = [...annotations].sort(
		(a, b) => a.timestamp - b.timestamp,
	);
	const { t } = useTranslation();
	return (
		<Collapsible open={historyOpen} onOpenChange={setHistoryOpen}>
			<ToolInvocation>
				<button
					type="button"
					onClick={() => setHistoryOpen(!historyOpen)}
					className="w-full cursor-pointer"
				>
					<ToolInvocationHeader>
						<ToolInvocationName name={titleResult} type="tool-result" />

						<span className="text-sm text-muted-foreground text-start">
							{historyOpen ? t("common.collapse") : t("common.expand")}
						</span>
					</ToolInvocationHeader>
				</button>
				<CollapsibleContent>
					<ToolInvocationContent className="px-0">
						<Separator />
						<ToolInvocationAnnotationsHistory annotations={sortedAnnotations} />
					</ToolInvocationContent>
				</CollapsibleContent>
			</ToolInvocation>
		</Collapsible>
	);
}

interface AnnotatedToolCallProps {
	titleCall: string;
	annotations: ToolAnnotation[];
}

function AnnotatedToolCall({ titleCall, annotations }: AnnotatedToolCallProps) {
	const [historyOpen, setHistoryOpen] = useState(false);
	const sortedAnnotations = [...annotations].sort(
		(a, b) => a.timestamp - b.timestamp,
	);
	const { t } = useTranslation();
	return (
		<Collapsible open={historyOpen} onOpenChange={setHistoryOpen}>
			<ToolInvocation>
				<button
					type="button"
					onClick={() => setHistoryOpen(!historyOpen)}
					className="w-full cursor-pointer"
				>
					<ToolInvocationHeader>
						<ToolInvocationName name={titleCall} type="tool-call" />

						{sortedAnnotations.length > 0 && (
							<>
								<ToolInvocationAnnotation
									annotation={sortedAnnotations[sortedAnnotations.length - 1]}
									className="gap-2"
									messageClassName="line-clamp-1 text-start"
								/>

								<span className="text-sm text-muted-foreground text-start">
									{historyOpen ? t("common.collapse") : t("common.expand")}
								</span>
							</>
						)}
					</ToolInvocationHeader>
				</button>
				<CollapsibleContent>
					<ToolInvocationContent className="px-0">
						<Separator />
						<ToolInvocationAnnotationsHistory annotations={sortedAnnotations} />
					</ToolInvocationContent>
				</CollapsibleContent>
			</ToolInvocation>
		</Collapsible>
	);
}

interface ToolInvocationAnnotationsHistoryProps {
	annotations: ToolAnnotation[];
}

function ToolInvocationAnnotationsHistory({
	annotations,
}: ToolInvocationAnnotationsHistoryProps) {
	return (
		<ScrollArea
			className={cn("flex flex-col mt-3 overflow-y-auto max-h-[16rem]")}
		>
			<div className={cn("flex flex-col gap-3 px-5")}>
				{annotations.map((annotation, index) => (
					<ToolInvocationAnnotation
						key={`${annotation.toolCallId}-${index}`}
						annotation={annotation}
					/>
				))}
			</div>
		</ScrollArea>
	);
}

const statusColorMap: Record<string, string> = {
	running: "bg-blue-500",
	completed: "bg-green-500",
	failed: "bg-red-500",
	default: "bg-gray-500",
};

interface ToolInvocationAnnotationProps {
	annotation: ToolAnnotation;
	className?: string;
	messageClassName?: string;
}

function ToolInvocationAnnotation({
	annotation,
	className,
	messageClassName,
}: ToolInvocationAnnotationProps) {
	const status = annotation.status;
	const statusColor = statusColorMap[status] || statusColorMap.default;

	const { t } = useTranslation();

	return (
		<div className={cn("flex items-start gap-3", className)}>
			<div
				className={cn("h-2 w-2 rounded-full shrink-0 mt-1.5", statusColor)}
			/>
			<div className={cn("text-sm", messageClassName)}>
				{t(annotation.messageKey)}
				{annotation.messageArgs && <span>: {annotation.messageArgs}</span>}
			</div>
		</div>
	);
}

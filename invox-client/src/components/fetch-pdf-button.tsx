import { useFetchPdf } from "@/hooks/use-fetch-pdf";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import {
	Popover,
	PopoverContent,
	PopoverTrigger,
} from "@/components/ui/popover";

export function FetchPdfButton({
	pdfId,
	pdfPage,
	pdfChunkIndex,
	children,
	className,
	asChild,
	download,
	filename,
}: {
	pdfId: string;
	pdfPage?: string;
	pdfChunkIndex?: string;
	children: React.ReactNode;
	className?: string;
	asChild?: boolean;
	download?: boolean;
	filename?: string;
}) {
	const { t } = useTranslation();
	const { isFetching, fetchPdf } = useFetchPdf();
	const [isPopoverOpen, setIsPopoverOpen] = useState(false);

	const handleFetchPdf = async () => {
		if (isFetching) return;

		setIsPopoverOpen(true);
		await fetchPdf({ pdfId, pdfPage, pdfChunkIndex, download, filename });
		setIsPopoverOpen(false);
	};

	const TriggerElement = asChild ? (children as React.ReactElement) : (
		<button
			type="button"
			className={cn(
				"cursor-pointer font-medium underline underline-offset-4",
				className,
			)}
			data-type="pdf-link"
			data-pdf-id={pdfId}
			data-pdf-page={pdfPage}
			data-pdf-chunk-index={pdfChunkIndex}
		>
			{children}
		</button>
	);

	return (
		<Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen}>
			<PopoverTrigger onClick={handleFetchPdf} asChild>
				{TriggerElement}
			</PopoverTrigger>
			<PopoverContent className="flex items-center p-2 gap-2">
				<Loader2 className="h-4 w-4 animate-spin text-primary" />
				<p className="text-sm">
					{t(
						"components.fetchPdfButton.retrievingDocument",
						"Retrieving document...",
					)}
				</p>
			</PopoverContent>
		</Popover>
	);
}


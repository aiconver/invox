import { useState } from "react";
import { useTranslation } from "react-i18next";
import { getArtifactPdf } from "@/services/artifacts";
import { toast } from "sonner";

export function useFetchPdf() {
	const { t } = useTranslation();
	const [isFetching, setIsFetching] = useState(false);

	const fetchPdf = async ({
		pdfId,
		pdfPage,
		pdfChunkIndex,
		download,
		filename,
	}: {
		pdfId: string;
		pdfPage?: string;
		pdfChunkIndex?: string;
		download?: boolean;
		filename?: string;
	}) => {
		if (isFetching) return;

		setIsFetching(true);

		try {
			const pdfBlob = await getArtifactPdf(pdfId, pdfChunkIndex);
			const pdfUrl = URL.createObjectURL(pdfBlob);

			const link = document.createElement("a");
			if (download) {
				link.href = pdfUrl;
				link.download = filename || `${pdfId}.pdf`;
				document.body.appendChild(link);
				link.click();
				document.body.removeChild(link);
			} else {
				link.href = pdfPage ? `${pdfUrl}#page=${Number(pdfPage)}` : pdfUrl;
				link.target = "_blank";
				link.rel = "noopener noreferrer";
				link.click();
			}

			setTimeout(() => URL.revokeObjectURL(pdfUrl), 5000); // Cleanup
		} catch (error) {
			console.error("Error fetching PDF:", {
				error,
				pdfId,
				pdfPage,
			});
			toast.error(t("common.error", "An unexpected error occurred."));
		} finally {
			setIsFetching(false);
		}
	};

	return { isFetching, fetchPdf };
}

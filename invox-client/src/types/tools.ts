export interface ToolAnnotation {
	toolCallId: string;
	status: "running" | "completed" | "failed";
	timestamp: number;
	messageKey: string;
	messageArgs?: string;
}

export interface ToolSource {
	documentId: string;
	documentName: string;
	documentPath: string;
	pageNumber: number;
}

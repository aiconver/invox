import {
	ChatInput,
	ChatInputSubmit,
	ChatInputTextArea,
	ChatInputToggle,
} from "@/components/ui/chat-input";
import {
	ChatMessage,
	ChatMessageAvatar,
	ChatMessageBody,
	ChatMessageContent,
	ChatMessageTimestamp,
} from "@/components/ui/chat-message";
import { ChatMessageArea } from "@/components/ui/chat-message-area";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useChat } from "@ai-sdk/react";
import { useAuth } from "react-oidc-context";
import { Navbar } from "@/components/layout/navbar";
import { ToolInvocationSourcesList } from "@/components/tools/sources-tool";
import { Separator } from "@/components/ui/separator";
import { AnnotatedTool } from "@/components/tools/annotated-tool";
import type { ToolAnnotation } from "@/types/tools";
import { FileExplorerPanel } from "@/components/file-explorer/file-explorer-panel";
import type { PathContentItem } from "@/types/path";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { X, FolderOpenIcon } from "lucide-react";
import { SelectedPathItems } from "@/components/file-explorer/selected-path-items";
import { toast } from "sonner";

const apiUrl = import.meta.env.VITE_APP_API_BASE_URL;

export function QuestionsAnswersPage() {
	const { t } = useTranslation();
	const { user } = useAuth();

	const [useDeepContentSearch, setUseDeepContentSearch] = useState(false);
	const [selectedFiles, setSelectedFiles] = useState<PathContentItem[]>([]);
	const [fileExplorerOpen, setFileExplorerOpen] = useState(false);

	const {
		messages,
		input,
		handleInputChange,
		handleSubmit,
		isLoading,
		stop,
		error,
	} = useChat({
		api: `${apiUrl}/artifacts/ai/chat/`,
		headers: {
			Authorization: `Bearer ${user?.access_token}`,
		},
		body: {
			useDeepContentSearch,
			selectedFiles: selectedFiles,
		},
		maxSteps: 5,
		onError: (error) => {
			console.error(error);
			toast.error(error.message.slice(0, 100));
		},
	});

	const handleFileSelect = (file: PathContentItem) => {
		setSelectedFiles((prevFiles) => {
			// Avoid adding duplicates
			if (
				prevFiles.some(
					(f) => f.path === file.path && f.serviceId === file.serviceId,
				)
			) {
				return prevFiles;
			}
			return [...prevFiles, file];
		});
	};

	const handleRemoveFile = (fileToRemove: PathContentItem) => {
		setSelectedFiles((prevFiles) =>
			prevFiles.filter(
				(file) =>
					file.path !== fileToRemove.path ||
					file.serviceId !== fileToRemove.serviceId,
			),
		);
	};

	const handleSubmitMessage = () => {
		if (isLoading) {
			return;
		}
		handleSubmit();
	};

	return (
		<main className="flex-1 overflow-auto">
			<div className="flex h-full overflow-hidden">
				<div className="flex-1 flex flex-col overflow-y-auto">
					<Navbar />
					<div className="flex-1 flex flex-col overflow-y-auto">
						<ChatMessageArea scrollButtonAlignment="center">
							<div className="max-w-2xl mx-auto w-full px-4 py-8 space-y-4">
								{messages.map((message) => {
									if (message.role !== "user") {
										return (
											<ChatMessage
												key={message.id}
												id={message.id}
												variant="default"
												type="incoming"
											>
												<ChatMessageAvatar />
												<ChatMessageBody>
													<ChatMessageTimestamp
														timestamp={
															message.createdAt
																? new Date(message.createdAt)
																: new Date()
														}
													/>
													{message.parts?.map((part, index) => {
														switch (part.type) {
															case "text":
																return (
																	<ChatMessageContent
																		key={`${message.id}-text-${index}`}
																		content={part.text}
																	/>
																);

															case "tool-invocation": {
																if (
																	(part.toolInvocation.toolName ===
																		"deepContentSearch" ||
																		part.toolInvocation.toolName ===
																			"contentSearch") &&
																	message.annotations
																) {
																	return (
																		<div className="flex flex-col gap-3 max-w-lg">
																			<AnnotatedTool
																				key={part.toolInvocation.toolCallId}
																				type={
																					"result" in part.toolInvocation
																						? "result"
																						: "call"
																				}
																				titleCall={t(
																					"components.toolInvocation.search.call.title",
																				)}
																				titleResult={t(
																					"components.toolInvocation.search.result.title",
																				)}
																				annotations={
																					message.annotations as unknown as ToolAnnotation[]
																				}
																			/>

																			{"result" in part.toolInvocation &&
																				part.toolInvocation.result && (
																					<>
																						<Separator />
																						<ToolInvocationSourcesList
																							sources={
																								part.toolInvocation.result
																									.sources
																							}
																							maxVisible={5}
																							maxHeight="10rem"
																						/>
																						{part.toolInvocation.result
																							.answer && (
																							<>
																								<Separator />
																								<ChatMessageContent
																									key={`${message.id}-text-${index}`}
																									content={
																										part.toolInvocation.result
																											.answer
																									}
																								/>
																							</>
																						)}
																					</>
																				)}
																		</div>
																	);
																}
															}
														}
													})}
												</ChatMessageBody>
											</ChatMessage>
										);
									}
									return (
										<ChatMessage
											key={message.id}
											id={message.id}
											variant="bubble"
											type="outgoing"
										>
											<ChatMessageBody>
												<ChatMessageTimestamp
													timestamp={
														message.createdAt
															? new Date(message.createdAt)
															: new Date()
													}
												/>
												{message.parts?.map((part, index) => {
													switch (part.type) {
														case "text":
															return (
																<ChatMessageContent
																	key={`${message.id}-text-${index}`}
																	content={part.text}
																/>
															);
													}
												})}
											</ChatMessageBody>
										</ChatMessage>
									);
								})}
								{isLoading && (
									<div className="flex space-x-2 justify-start items-center ml-16 pt-6">
										<div className="h-2 w-2 bg-muted-foreground rounded-full animate-bounce" />
										<div className="h-2 w-2 bg-muted-foreground rounded-full animate-bounce delay-100" />
										<div className="h-2 w-2 bg-muted-foreground rounded-full animate-bounce delay-200" />
									</div>
								)}
							</div>
						</ChatMessageArea>
						<div className="px-2 py-4 max-w-2xl mx-auto w-full">
							{selectedFiles.length > 0 && (
								<div className="mb-2 px-3">
									<div className="text-sm text-muted-foreground mb-1">
										{t("components.selectedFilesLabel")}
									</div>
									<SelectedPathItems
										items={selectedFiles}
										onRemove={handleRemoveFile}
									/>
								</div>
							)}
							<ChatInput
								value={input}
								onChange={handleInputChange}
								onSubmit={handleSubmitMessage}
								loading={isLoading}
								onStop={stop}
							>
								<ChatInputTextArea placeholder={t("common.submit")} />
								<div className="flex justify-between w-full">
									<div className="flex">
										<ChatInputToggle
											useDeepContentSearch={useDeepContentSearch}
											onUseDeepContentSearchChange={setUseDeepContentSearch}
										/>
										<Separator
											orientation="vertical"
											className="h-[70%] my-auto ml-3 mr-1"
										/>
										<Button
											variant="ghost"
											size="icon"
											className="shrink-0"
											onClick={() => setFileExplorerOpen(true)}
										>
											<FolderOpenIcon />
											<span className="sr-only">Open File Explorer</span>
										</Button>
									</div>
									<ChatInputSubmit />
								</div>
							</ChatInput>
						</div>
					</div>
				</div>
				{/* File Explorer Panel (side by side) */}
				{fileExplorerOpen && (
					<FileExplorerPanel
						open={fileExplorerOpen}
						onClose={() => setFileExplorerOpen(false)}
						selectedItems={selectedFiles}
						onAddToContext={handleFileSelect}
						onRemoveFromContext={handleRemoveFile}
					/>
				)}
			</div>
		</main>
	);
}

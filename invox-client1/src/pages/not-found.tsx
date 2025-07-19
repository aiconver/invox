import { useTranslation } from "react-i18next";

export function NotFound() {
	const { t } = useTranslation();

	return (
		<div className="h-screen w-full flex flex-col items-center justify-center">
			<div className="flex flex-col items-center space-y-4">
				<div className="text-9xl font-bold text-muted-foreground">404</div>

				<h1 className="text-2xl font-semibold tracking-tight">
					{t("pages.notFound.title")}
				</h1>

				<p className="text-muted-foreground text-center max-w-[500px]">
					{t("pages.notFound.description")}
				</p>
			</div>
		</div>
	);
}

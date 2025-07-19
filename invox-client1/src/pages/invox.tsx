import { Navbar } from "@/components/layout/navbar";
import { useTranslation } from "react-i18next";

export function Invox() {
	const { t } = useTranslation();

	return (
		<main className="flex-1 overflow-auto">
			<div className="flex h-full overflow-hidden">
				<div className="flex-1 flex flex-col overflow-y-auto">
					<Navbar />
				</div>
			</div>
		</main>
	);
}

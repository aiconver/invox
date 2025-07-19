import { Navbar } from "@/components/layout/navbar";
import { MsGraphSelectResources } from "../../components/settings/credentials/ms-graph-select-resources";

export function CredentialsMsGraphSelectResourcesPage() {
	return (
		<main className="flex flex-col h-full">
			<Navbar />
			<MsGraphSelectResources />
		</main>
	);
}

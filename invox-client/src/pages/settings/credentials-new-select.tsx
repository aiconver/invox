import { Navbar } from "@/components/layout/navbar";
import { CredentialsSelect } from "@/components/settings/credentials/credentials-select";

export function CredentialsNewSelectPage() {
	return (
		<main className="flex flex-col h-full">
			<Navbar />
			<CredentialsSelect />
		</main>
	);
}

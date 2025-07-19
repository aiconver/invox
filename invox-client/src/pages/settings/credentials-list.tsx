import { Navbar } from "@/components/layout/navbar";
import { CredentialsList } from "@/components/settings/credentials/credentials-list";

export function CredentialsListPage() {
	return (
		<main className="flex flex-col h-full">
			<Navbar />
			<CredentialsList />
		</main>
	);
}

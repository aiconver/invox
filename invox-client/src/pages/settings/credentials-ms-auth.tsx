import { Navbar } from "@/components/layout/navbar";
import { MicrosoftAuth } from "@/components/settings/credentials/microsoft-auth";

export function CredentialsMsAuthPage() {
	return (
		<main className="flex flex-col h-full">
			<Navbar />
			<MicrosoftAuth />
		</main>
	);
}

import { Navbar } from "@/components/layout/navbar";
import { CredentialsAdd } from "@/components/settings/credentials/credentials-add";
import { serviceConfigs } from "@/lib/service-config";
import { useParams } from "react-router-dom";
import { NotFound } from "../not-found";

export function CredentialsAddPage() {
	const { serviceType } = useParams<{ serviceType: "samba" | "webdav" }>();

	if (!serviceType || !Object.keys(serviceConfigs).includes(serviceType)) {
		return <NotFound />;
	}

	return (
		<main className="flex flex-col h-full">
			<Navbar />
			<CredentialsAdd service={serviceType} />
		</main>
	);
}

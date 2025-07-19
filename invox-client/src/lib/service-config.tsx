import { Server, HardDrive } from "lucide-react";
import { MicrosoftIcon } from "@/components/icons/microsoft";
import type { ServiceType } from "@/types/credentials";

export interface ServiceConfig {
	key: ServiceType;
	icon: React.ElementType;
	title: string;
	description: string;
}

export const serviceConfigs: Record<ServiceType, ServiceConfig> = {
	samba: {
		key: "samba",
		icon: Server,
		title: "Samba",
		description: "Connect to Samba/SMB file shares",
	},
	webdav: {
		key: "webdav",
		icon: HardDrive,
		title: "WebDAV",
		description: "Connect to WebDAV servers",
	},
	ms_graph: {
		key: "ms_graph",
		icon: MicrosoftIcon,
		title: "Microsoft 365",
		description: "Connect to Microsoft 365 / OneDrive",
	},
};

// Helper functions for backward compatibility
export const getServiceIcon = (service: ServiceType | string) => {
	return serviceConfigs[service as ServiceType]?.icon || HardDrive;
};

export const getServiceLabel = (service: ServiceType | string) => {
	return serviceConfigs[service as ServiceType]?.title || service;
};

export const getServiceDescription = (service: ServiceType | string) => {
	return serviceConfigs[service as ServiceType]?.description || 'Unknown service description';
};

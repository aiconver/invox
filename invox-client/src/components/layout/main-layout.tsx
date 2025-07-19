import { Outlet } from "react-router-dom";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";

export function MainLayout() {
	return (
		<SidebarProvider>
			<AppSidebar />
			<SidebarInset className="flex flex-col h-screen overflow-hidden min-h-0">
				<Outlet />
			</SidebarInset>
		</SidebarProvider>
	);
}

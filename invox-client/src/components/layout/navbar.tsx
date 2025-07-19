import { DynamicBreadcrumb } from "@/components/layout/dynamic-breadcrumb";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";

export function Navbar() {
	return (
		<header className="sticky top-0 flex h-14 shrink-0 items-center gap-2 bg-background">
			<div className="flex flex-1 items-center gap-2 px-3">
				<SidebarTrigger />
				<Separator orientation="vertical" className="mr-2 h-4" />
				<DynamicBreadcrumb />
			</div>
		</header>
	);
}

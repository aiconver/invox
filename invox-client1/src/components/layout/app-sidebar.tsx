import {
	Sidebar,
	SidebarContent,
	SidebarFooter,
	SidebarGroup,
	SidebarGroupContent,
	SidebarGroupLabel,
	SidebarHeader,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
} from "@/components/ui/sidebar";
import { APP_ROUTES } from "@/lib/routes";
import type { AppRoute } from "@/types/routes";
import { useTranslation } from "react-i18next";
import { MdChevronRight, MdOutlinePerson } from "react-icons/md";
import { Link, useLocation } from "react-router-dom";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "react-oidc-context";

export function AppSidebar() {
	const { t } = useTranslation();
	const location = useLocation();
	// const { user, signoutRedirect } = useAuth();

	const availableRoutes: AppRoute[] = [];

	availableRoutes.push(
		APP_ROUTES["invox"],
	);

	const isRouteActive = (route: AppRoute) => {
		const fullPath = location.pathname + location.search;
		const regex = new RegExp(route.pathPattern);
		return regex.test(fullPath);
	};

	return (
		<Sidebar>
			<SidebarHeader className="items-center">
				<img src="/qa/logo.svg" alt="logo" className="size-10" />
			</SidebarHeader>
			<SidebarContent>
				<SidebarGroup>
					<SidebarGroupLabel>AI Conver</SidebarGroupLabel>
					<SidebarGroupContent>
						<SidebarMenu>
							{availableRoutes.map((route) => {
								const { to, label, icon: Icon } = route;
								return (
									<SidebarMenuItem key={to}>
										<SidebarMenuButton
											asChild
											isActive={isRouteActive(route)}
										>
											<Link to={to}>
												{Icon && <Icon />}
												<span>{t(label)}</span>
											</Link>
										</SidebarMenuButton>
									</SidebarMenuItem>
								);
							})}
						</SidebarMenu>
					</SidebarGroupContent>
				</SidebarGroup>
			</SidebarContent>
			<SidebarFooter>
				<SidebarMenu>
					<SidebarMenuItem>
						<DropdownMenu>
							{/* <DropdownMenuTrigger asChild>
								<SidebarMenuButton>
									<MdOutlinePerson /> {user?.profile.name}
									<MdChevronRight className="ml-auto rotate-90" />
								</SidebarMenuButton>
							</DropdownMenuTrigger> */}
							{/* <DropdownMenuContent
								side="top"
								className="w-[--radix-popper-anchor-width]"
							>
								<DropdownMenuItem onClick={() => void signoutRedirect()}>
									<span>{t("common.signOut")}</span>
								</DropdownMenuItem>
							</DropdownMenuContent> */}
						</DropdownMenu>
					</SidebarMenuItem>
				</SidebarMenu>
			</SidebarFooter>
		</Sidebar>
	);
}

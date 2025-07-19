import {
	Breadcrumb,
	BreadcrumbItem,
	BreadcrumbList,
	BreadcrumbPage,
	BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { APP_ROUTES } from "@/lib/routes";
import { useTranslation } from "react-i18next";
import { useLocation } from "react-router-dom";

export function DynamicBreadcrumb() {
	const { t } = useTranslation();
	const location = useLocation();

	const currentPathRoute = Object.values(APP_ROUTES).find(
		(route) => route.to === location.pathname,
	);

	if (!currentPathRoute) {
		return null;
	}

	return (
		<Breadcrumb>
			<BreadcrumbList>
				<BreadcrumbItem>
					<BreadcrumbPage>{t(currentPathRoute.label)}</BreadcrumbPage>
				</BreadcrumbItem>
			</BreadcrumbList>
		</Breadcrumb>
	);
}

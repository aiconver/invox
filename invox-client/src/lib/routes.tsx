import type { AppRoutes } from "@/types/routes";
import { Bot } from "lucide-react";

export const APP_ROUTES: AppRoutes = {
	"invox": {
		label: "navigation.invox",
		to: "/qa/",
		icon: Bot,
		pathPattern: "^/qa/$",
	},
} as const;

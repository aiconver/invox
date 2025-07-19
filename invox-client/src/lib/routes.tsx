import type { AppRoutes } from "@/types/routes";
import { KeyRound, FileText, Video } from "lucide-react";
import { MdQuestionAnswer, MdSettings } from "react-icons/md";

export const APP_ROUTES: AppRoutes = {
	"about": {
		label: "navigation.about",
		to: "/qa/",
		icon: FileText,
		pathPattern: "^/qa/about/$",
	},
} as const;

import type { AppRoutes } from "@/types/routes";
import { KeyRound, FileText, Video } from "lucide-react";
import { MdQuestionAnswer, MdSettings } from "react-icons/md";

export const APP_ROUTES: AppRoutes = {
	"questions-answers": {
		label: "navigation.questionsAnswers",
		to: "/qa/questions-answers/",
		icon: MdQuestionAnswer,
		pathPattern: "^/qa/questions-answers/$",
	},
	"artifacts": {
		label: "navigation.artifacts",
		to: "/qa/artifacts/",
		icon: FileText,
		pathPattern: "^/qa/artifacts/$",
	},
	"about": {
		label: "navigation.about",
		to: "/qa/",
		icon: FileText,
		pathPattern: "^/qa/about/$",
	},
	"recordings": {
		label: "navigation.recordings",
		to: "/qa/artifacts/?fileType=Videos&fileType=Audio",
		icon: Video,
		pathPattern: "^/qa/artifacts/\\?.*fileType=Videos&fileType=Audio",
	},
	"settings": {
		label: "navigation.settings",
		to: "/qa/settings/",
		icon: MdSettings,
		pathPattern: "^/qa/settings/",
	},
	"settings-credentials-list": {
		label: "navigation.settingsCredentialsList",
		to: "/qa/settings/credentials/list/",
		icon: MdSettings,
		pathPattern: "^/qa/settings/credentials/list/$",
	},
	"settings-credentials-new-select": {
		label: "navigation.settingsCredentialsNewSelect",
		to: "/qa/settings/credentials/new/select/",
		icon: MdSettings,
		pathPattern: "^/qa/settings/credentials/new/select/$",
	},
	"settings-credentials-new-form": {
		label: "navigation.settingsCredentialsNewForm",
		to: "/qa/settings/credentials/new",
		icon: MdSettings,
		pathPattern: "^/qa/settings/credentials/new/[^/]+$",
	},
	"settings-credentials-new-ms-auth": {
		label: "navigation.settingsCredentialsMsAuth",
		to: "/qa/settings/credentials/new/ms-auth/",
		icon: MdSettings,
		pathPattern: "^/qa/settings/credentials/new/ms-auth/$",
	},
	"settings-credentials-ms-graph-select-resources": {
		label: "navigation.settingsCredentialsMsGraphSelectResources",
		to: "/qa/settings/credentials/new/ms-graph-select-resources/:serviceAuthId/",
		icon: MdSettings,
		pathPattern: "^/qa/settings/credentials/new/ms-graph-select-resources/[^/]+/$",
	},
	"settings-credentials-error": {
		label: "navigation.settingsCredentialsError",
		to: "/qa/settings/credentials/error/",
		icon: MdSettings,
		pathPattern: "^/qa/settings/credentials/error/$",
	},
} as const;

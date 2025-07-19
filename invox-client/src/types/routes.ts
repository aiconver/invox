export type AppRoute = {
	label: string;
	to: string;
	icon?: React.ElementType;
	pathPattern: string;
};

const routes = [
	"questions-answers",
	"artifacts",
	"recordings",
	"settings",
	"about",
	"settings-credentials-list",
	"settings-credentials-new-select",
	"settings-credentials-new-form",
	"settings-credentials-new-ms-auth",
	"settings-credentials-ms-graph-select-resources",
	"settings-credentials-error",
] as const;

type Route = (typeof routes)[number];

export type AppRoutes = Record<Route, AppRoute>;

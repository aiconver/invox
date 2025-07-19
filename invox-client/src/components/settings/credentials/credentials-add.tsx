import { Button } from "@/components/ui/button";

import { toast } from "sonner";
import { Loader2 } from "lucide-react";
import { createCredential } from "@/services/credentials";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Form } from "@/components/ui/form";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
	FormField,
	FormItem,
	FormLabel,
	FormControl,
	FormMessage,
	FormDescription,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { SERVICE_TYPES, type ServiceType } from "@/types/credentials";
import { getServiceLabel } from "@/lib/service-config";
import { Link, useNavigate } from "react-router-dom";
import { APP_ROUTES } from "@/lib/routes";

const ServiceCredentialSchema = z.object({
	service: z.enum(SERVICE_TYPES),
	user: z.string().min(1, { message: "User is required." }),
	pass: z.string().min(1, { message: "Password is required." }),
	host: z.string().min(1, { message: "Host is required." }),
	settings: z.string().optional(),
});

export function CredentialsAdd({
	service,
}: {
	service: "samba" | "webdav";
}) {
	const { t } = useTranslation();
	const navigate = useNavigate();

	const queryClient = useQueryClient();

	const form = useForm<z.infer<typeof ServiceCredentialSchema>>({
		resolver: zodResolver(ServiceCredentialSchema),
		defaultValues: {
			service: service,
			user: "",
			pass: "",
			host: "",
		},
	});

	const { mutate: createCredentialMutation, isPending } = useMutation({
		mutationFn: createCredential,
		onSuccess: () => {
			toast.success(t("components.settings.credentials.add.success"));
			queryClient.invalidateQueries({ queryKey: ["credentials"] });
			navigate(APP_ROUTES["settings-credentials-list"].to);
		},
		onError: () => {
			toast.error(t("components.settings.credentials.add.error"));
		},
	});

	const onSubmit = (data: z.infer<typeof ServiceCredentialSchema>) => {
		createCredentialMutation(data);
	};

	const selectedService = form.watch("service");

	return (
		<Form {...form}>
			<form
				onSubmit={form.handleSubmit(onSubmit)}
				className="flex flex-col h-full p-8"
			>
				<div className="flex items-center justify-between mb-4">
					<h3 className="font-medium text-lg">{getServiceLabel(service)}</h3>
					<div className="flex items-center gap-2">
						<Button type="button" size="sm" variant="outline" asChild>
							<Link to={APP_ROUTES["settings-credentials-new-select"].to}>
								{t("common.cancel", "Cancel")}
							</Link>
						</Button>
						<Button type="submit" disabled={isPending} size="sm">
							{isPending ? (
								<Loader2 className="w-4 h-4 animate-spin" />
							) : (
								t("common.submit", "Submit")
							)}
						</Button>
					</div>
				</div>
				<div className="flex-1 overflow-y-auto pr-2 space-y-2">
					<div className="flex flex-col gap-2">
						<FormField
							control={form.control}
							name="host"
							render={({ field }) => (
								<FormItem>
									<FormLabel>
										{t(
											"components.settings.credentials.add.fields.host.label",
											"Host / URL",
										)}
									</FormLabel>
									<FormControl>
										<Input
											placeholder={t(
												"components.settings.credentials.add.fields.host.placeholder",
												"e.g., 192.168.1.100 or mydomain.com/webdav",
											)}
											{...field}
										/>
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>

						<FormField
							control={form.control}
							name="user"
							render={({ field }) => (
								<FormItem>
									<FormLabel>
										{t(
											"components.settings.credentials.add.fields.user.label",
											"Username",
										)}
									</FormLabel>
									<FormControl>
										<Input
											placeholder={t(
												"components.settings.credentials.add.fields.user.placeholder",
												"Enter username",
											)}
											{...field}
										/>
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>

						<FormField
							control={form.control}
							name="pass"
							render={({ field }) => (
								<FormItem>
									<FormLabel>
										{t(
											"components.settings.credentials.add.fields.pass.label",
											"Password",
										)}
									</FormLabel>
									<FormControl>
										<Input
											type="password"
											placeholder={t(
												"components.settings.credentials.add.fields.pass.placeholder",
												"Enter password",
											)}
											{...field}
										/>
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>

						{selectedService === "samba" && (
							<FormField
								control={form.control}
								name="settings"
								render={({ field }) => (
									<FormItem>
										<FormLabel>
											{t(
												"components.settings.credentials.add.fields.settings.label",
												"Share Path / Settings",
											)}
										</FormLabel>
										<FormControl>
											<Input
												placeholder={t(
													"components.settings.credentials.add.fields.settings.placeholder",
													"e.g., /share or workgroup=WORKGROUP",
												)}
												{...field}
												value={field.value || ""}
											/>
										</FormControl>
										<FormDescription>
											{t(
												"components.settings.credentials.add.fields.settings.description",
												"For Samba, this can be the share name (e.g., 'documents') or additional smbclient parameters.",
											)}
										</FormDescription>
										<FormMessage />
									</FormItem>
								)}
							/>
						)}
					</div>
				</div>
			</form>
		</Form>
	);
}

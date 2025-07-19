import { cn } from "@/lib/utils";
import { MessageCircleWarningIcon } from "lucide-react";
import { Card } from "@/components/ui/card";

interface AuthErrorProps {
	title: string;
	description: string;
	className?: string;
}

export function AuthError({ title, description, className }: AuthErrorProps) {
	return (
		<div className="h-screen flex items-center justify-center p-4">
			<Card
				className={cn(
					"max-w-md text-center space-y-4 p-6 rounded-lg",
					"text-destructive",
					className,
				)}
			>
				<MessageCircleWarningIcon className="h-12 w-12 mx-auto" />
				<h2 className="text-2xl font-semibold">{title}</h2>
				{description && <p className="text-muted-foreground">{description}</p>}
			</Card>
		</div>
	);
}

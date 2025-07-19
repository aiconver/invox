import { Skeleton } from "@/components/ui/skeleton";

export function LayoutSkeleton() {
	return (
		<div className="flex min-h-screen w-full flex-col">
			<header className="sticky top-0 z-50 border-b bg-background">
				<div className="container mx-auto flex h-16 items-center px-4 md:px-10">
					{/* Logo Skeleton */}
					<Skeleton className="h-8 w-24" />

					<div className="ml-auto flex items-center space-x-4">
						<Skeleton className="h-10 w-24" />
						<Skeleton className="h-10 w-24" />
						<Skeleton className="h-10 w-24" />
					</div>
				</div>
			</header>

			<main className="container mx-auto flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
				<div className="space-y-4">
					<Skeleton className="h-12 w-[250px]" />
					<div className="flex gap-4">
						<Skeleton className="h-24 flex-1" />
						<Skeleton className="h-24 flex-1" />
						<Skeleton className="h-24 flex-1" />
					</div>
				</div>

				<div className="mt-8 space-y-4">
					<Skeleton className="h-8 w-[200px]" />
					<div className="grid gap-4 md:grid-cols-2">
						<Skeleton className="h-32" />
						<Skeleton className="h-32" />
					</div>
				</div>
			</main>
		</div>
	);
}

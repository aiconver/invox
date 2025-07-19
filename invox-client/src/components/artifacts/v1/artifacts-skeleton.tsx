import { Skeleton } from "@/components/ui/skeleton";

// Header skeleton that mimics ArtifactsHeader
export function ArtifactsHeaderSkeleton() {
	return (
		<div className="p-4 border-b">
			<div className="flex items-center justify-between gap-4">
				<div className="flex-1 relative">
					{/* Search input skeleton */}
					<Skeleton className="h-10 w-full" />
				</div>
			</div>
			{/* Results count skeleton */}
			<Skeleton className="h-4 w-32 mt-2 ml-2" />
		</div>
	);
}

// Enhanced list item skeleton that mimics the new ArtifactListItem
export function ArtifactListItemSkeleton() {
	return (
		<div className="w-full border rounded-xl bg-card shadow">
			<div className="p-4">
				<div className="flex gap-6">
					{/* Enhanced preview image skeleton */}
					<div className="flex-shrink-0">
						<Skeleton className="w-64 h-72 rounded border" />
					</div>

					{/* Content section skeleton */}
					<div className="flex-1 min-w-0 space-y-4">
						{/* Title skeleton */}
						<div>
							<Skeleton className="h-6 w-full mb-3" />
							<Skeleton className="h-6 w-4/5 mb-3" />
						</div>

						{/* Summary sections skeleton */}
						<div className="bg-blue-50 rounded-lg p-4">
							<Skeleton className="h-4 w-24 mb-2" />
							<Skeleton className="h-4 w-full mb-1" />
							<Skeleton className="h-4 w-full mb-1" />
							<Skeleton className="h-4 w-3/4" />
						</div>

						<div className="bg-gray-50 rounded-lg p-4">
							<Skeleton className="h-4 w-32 mb-2" />
							<Skeleton className="h-4 w-full mb-1" />
							<Skeleton className="h-4 w-full mb-1" />
							<Skeleton className="h-4 w-full mb-1" />
							<Skeleton className="h-4 w-2/3" />
						</div>

						{/* Quick stats skeleton */}
						<div className="flex flex-wrap gap-3">
							<div className="flex items-center gap-1">
								<Skeleton className="w-4 h-4" />
								<Skeleton className="h-4 w-20" />
							</div>
							<div className="flex items-center gap-1">
								<Skeleton className="w-4 h-4" />
								<Skeleton className="h-4 w-16" />
							</div>
							<div className="flex items-center gap-1">
								<Skeleton className="w-4 h-4" />
								<Skeleton className="h-4 w-18" />
							</div>
						</div>
					</div>
				</div>

				{/* Action buttons skeleton */}
				<div className="mt-4 pt-4 border-t">
					<div className="flex items-center justify-between">
						<div className="flex gap-2">
							<Skeleton className="h-8 w-20" />
							<Skeleton className="h-8 w-24" />
							<Skeleton className="h-8 w-16" />
						</div>
						<div className="flex gap-2">
							<Skeleton className="h-8 w-16" />
							<Skeleton className="h-8 w-20" />
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}

// Main artifacts skeleton that only shows list view
export function ArtifactsSkeleton({ viewMode }: { viewMode: "list" }) {
	return (
		<div className="flex flex-col h-full">
			<ArtifactsHeaderSkeleton />

			<div className="flex-1 overflow-hidden">
				<div className="p-6">
					<div className="space-y-4 max-w-screen-xl mx-auto">
						{Array.from({ length: 6 }).map((_, index) => (
							<ArtifactListItemSkeleton key={index} />
						))}
					</div>
				</div>
			</div>
		</div>
	);
} 
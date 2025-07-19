import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import type { PathContentItem } from "@/types/path";

interface SelectedPathItemsProps {
	items: PathContentItem[];
	onRemove: (item: PathContentItem) => void;
}

export function SelectedPathItems({ items, onRemove }: SelectedPathItemsProps) {
	return (
		<div className="flex flex-wrap gap-2">
			{items.map((item) => (
				<Badge
					key={`${item.serviceId}-${item.path}`}
					variant="secondary"
					className="flex items-center gap-1 max-w-[180px] truncate text-sm"
				>
					<span className="truncate" title={item.name}>
						{item.name}
					</span>
					<Button
						variant="ghost"
						size="icon"
						className="h-4 w-4 ml-1 -mr-1"
						onClick={() => onRemove(item)}
						title={item.name || "Remove from context"}
					>
						<span className="sr-only">Remove from context</span>
						<X className="h-3 w-3" />
					</Button>
				</Badge>
			))}
		</div>
	);
}

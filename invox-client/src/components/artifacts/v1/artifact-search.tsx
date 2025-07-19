import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface ArtifactSearchProps {
	onSearch: (query: string) => void;
	initialValue?: string;
	placeholder?: string;
}

export function ArtifactSearch({ onSearch, initialValue = "", placeholder }: ArtifactSearchProps) {
	const { t } = useTranslation();
	const [query, setQuery] = useState(initialValue);

	// Debounce search
	useEffect(() => {
		const timer = setTimeout(() => {
			onSearch(query);
		}, 300);

		return () => clearTimeout(timer);
	}, [query, onSearch]);

	const handleClear = () => {
		setQuery("");
	};

	return (
		<div className="relative">
			<div className="relative">
				<Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
				<Input
					type="text"
					value={query}
					onChange={(e) => setQuery(e.target.value)}
					placeholder={placeholder || t("components.artifacts.searchPlaceholder")}
					className="pl-10 pr-10"
				/>
				{query && (
					<Button
						variant="ghost"
						size="sm"
						onClick={handleClear}
						className="absolute right-1 top-1/2 transform -translate-y-1/2 h-8 w-8 p-0 hover:bg-muted"
					>
						<X className="w-4 h-4" />
					</Button>
				)}
			</div>
		</div>
	);
} 
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

export function getUniqueItems<T>(arr: T[], cb: (item: T) => string): T[] {
	const hashTable: Record<string, T> = {};
	for (let i = 0; i < arr.length; i++) {
		const item = arr[i];
		if (!hashTable[cb(item)]) {
			hashTable[cb(item)] = item;
		}
	}
	return Object.values(hashTable);
}

/**
 * Convert a string like "LayoutSpacing" to "Layout Spacing"
 * @param text
 * @returns
 */
export function pascalToSentence(text: string): string {
	return (
		text
			// Add space before capital letters
			.replace(/([A-Z])/g, " $1")
			// Trim extra spaces and capitalize first letter
			.trim()
	);
}

/**
 * Convert a string like "layout-spacing" to "Layout Spacing"
 * @param idToConvert
 * @returns
 */
export const idToReadableText = (idToConvert: string) => {
	return idToConvert
		.split("-")
		.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
		.join(" ");
};

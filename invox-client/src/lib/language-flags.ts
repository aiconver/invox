// Language to country code mapping for flags
export const LANGUAGE_TO_FLAG_MAP: Record<string, string> = {
	en: 'gb', // English -> Great Britain flag
	de: 'de', // German -> Germany flag
	fr: 'fr', // French -> France flag
	es: 'es', // Spanish -> Spain flag
	pt: 'pt', // Portuguese -> Portugal flag
	it: 'it', // Italian -> Italy flag
	nl: 'nl', // Dutch -> Netherlands flag
	sv: 'se', // Swedish -> Sweden flag
	da: 'dk', // Danish -> Denmark flag
	no: 'no', // Norwegian -> Norway flag
	fi: 'fi', // Finnish -> Finland flag
	pl: 'pl', // Polish -> Poland flag
	ru: 'ru', // Russian -> Russia flag
	zh: 'cn', // Chinese -> China flag
	ja: 'jp', // Japanese -> Japan flag
	ko: 'kr', // Korean -> South Korea flag
	ar: 'sa', // Arabic -> Saudi Arabia flag
};

/**
 * Get the flag country code for a given language code
 */
export function getLanguageFlag(languageCode: string): string {
	const normalizedCode = languageCode.toLowerCase().slice(0, 2);
	return LANGUAGE_TO_FLAG_MAP[normalizedCode] || 'us'; // Default to US flag
}

/**
 * Get the flag image path for a language
 */
export function getLanguageFlagPath(languageCode: string): string {
	const flagCode = getLanguageFlag(languageCode);
	return `/flags/1x1/${flagCode}.svg`;
}

/**
 * Get a readable language name from language code
 */
export function getLanguageName(languageCode: string): string {
	const names: Record<string, string> = {
		en: 'English',
		de: 'Deutsch',
		fr: 'Français',
		es: 'Español',
		pt: 'Português',
		it: 'Italiano',
		nl: 'Nederlands',
		sv: 'Svenska',
		da: 'Dansk',
		no: 'Norsk',
		fi: 'Suomi',
		pl: 'Polski',
		ru: 'Русский',
		zh: '中文',
		ja: '日本語',
		ko: '한국어',
		ar: 'العربية',
	};
	
	const normalizedCode = languageCode.toLowerCase().slice(0, 2);
	return names[normalizedCode] || languageCode.toUpperCase();
} 
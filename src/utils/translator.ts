/**
 * Reader bilingual translator (Phase 1: hardcoded local llama-server backend).
 *
 * State machine: 'off' → 'bilingual' (原文+繁中) → 'target_only' (全繁中) → 'off'
 *
 * Design:
 * - Article paragraphs are batched (BATCH_SIZE per request) to amortize
 *   prompt overhead.
 * - In-memory translation cache keyed by SHA256(text + targetLang) survives
 *   state toggles within the same reader session.
 * - DOM mutation is non-destructive: original text is preserved as
 *   `data-ot-original` so we can restore on 'off' / re-render on toggle.
 *
 * Backend defaults to local Qwen3-4B-Instruct on http://127.0.0.1:8843,
 * configurable via reader settings (translatorEndpoint / translatorModel).
 */

const BATCH_SIZE = 5;
const SEPARATOR = '\n###|||###\n';
const TRANSLATABLE_SELECTOR = 'article > p, article > h2, article > h3, article > h4, article > h5, article > h6, article > li, article > blockquote';
const ORIG_ATTR = 'data-ot-original';
const TRANS_ATTR = 'data-ot-translated';
const BILINGUAL_CLASS = 'ot-bilingual-translated';
const HIDDEN_ORIG_CLASS = 'ot-original-hidden';
const TRANSLATING_CLASS = 'ot-translating';

export type ReaderTranslateState = 'off' | 'bilingual' | 'target_only';

export interface TranslatorConfig {
	endpoint: string;        // e.g. http://127.0.0.1:8843/v1/chat/completions
	model: string;           // e.g. qwen3-4b-translator
	targetLang: string;      // human label, e.g. "繁體中文（台灣）"
	systemPrompt: string;    // overrideable
	timeoutMs: number;
}

export const DEFAULT_TRANSLATOR_CONFIG: TranslatorConfig = {
	endpoint: 'http://127.0.0.1:8843/v1/chat/completions',
	model: 'qwen3-4b-translator',
	targetLang: '繁體中文（台灣）',
	systemPrompt: [
		'你是專業翻譯助手。將使用者提供的多段文本翻譯成繁體中文（台灣慣用語）。',
		'規則：',
		`1. 段落之間以 "${SEPARATOR.trim()}" 分隔，請維持相同分隔符與段落數`,
		'2. 只輸出翻譯，不要任何前後說明、不要 markdown wrapper',
		'3. 保留原文的標點與換行節奏',
		'4. 專有名詞、code、URL、數字保持原樣不譯',
		'5. 已是繁體中文的段落原樣輸出',
	].join('\n'),
	timeoutMs: 60000,
};

const cache = new Map<string, string>();

async function sha256(input: string): Promise<string> {
	const buf = new TextEncoder().encode(input);
	const hash = await crypto.subtle.digest('SHA-256', buf);
	return Array.from(new Uint8Array(hash))
		.map(b => b.toString(16).padStart(2, '0'))
		.join('');
}

async function cacheKey(text: string, lang: string): Promise<string> {
	return `${lang}::${await sha256(text)}`;
}

async function translateBatch(
	texts: string[],
	cfg: TranslatorConfig,
): Promise<string[]> {
	if (texts.length === 0) return [];
	const userContent = texts.join(SEPARATOR);
	const ctrl = new AbortController();
	const timeout = setTimeout(() => ctrl.abort(), cfg.timeoutMs);
	try {
		const resp = await fetch(cfg.endpoint, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model: cfg.model,
				messages: [
					{ role: 'system', content: cfg.systemPrompt },
					{ role: 'user', content: userContent },
				],
				temperature: 0.3,
				stream: false,
			}),
			signal: ctrl.signal,
		});
		if (!resp.ok) throw new Error(`translator HTTP ${resp.status}`);
		const json = await resp.json();
		const out: string = json?.choices?.[0]?.message?.content ?? '';
		const parts = out.split(SEPARATOR).map(p => p.trim());
		// Heal mismatched batch counts (rare model misbehaviour): pad/truncate
		if (parts.length < texts.length) {
			while (parts.length < texts.length) parts.push('');
		} else if (parts.length > texts.length) {
			parts.length = texts.length;
		}
		return parts;
	} finally {
		clearTimeout(timeout);
	}
}

export async function translateTexts(
	texts: string[],
	cfg: TranslatorConfig = DEFAULT_TRANSLATOR_CONFIG,
): Promise<string[]> {
	const out: string[] = new Array(texts.length).fill('');
	const need: { idx: number; text: string }[] = [];

	// Cache check
	for (let i = 0; i < texts.length; i++) {
		const key = await cacheKey(texts[i], cfg.targetLang);
		const hit = cache.get(key);
		if (hit !== undefined) {
			out[i] = hit;
		} else {
			need.push({ idx: i, text: texts[i] });
		}
	}

	// Batch the misses
	for (let i = 0; i < need.length; i += BATCH_SIZE) {
		const slice = need.slice(i, i + BATCH_SIZE);
		const translations = await translateBatch(slice.map(s => s.text), cfg);
		for (let j = 0; j < slice.length; j++) {
			const t = translations[j] || '';
			out[slice[j].idx] = t;
			const key = await cacheKey(slice[j].text, cfg.targetLang);
			if (t) cache.set(key, t);
		}
	}

	return out;
}

/**
 * Apply translation state to the article.
 * Idempotent — calling repeatedly with the same state is a no-op for already-rendered nodes.
 */
export async function applyTranslationState(
	article: HTMLElement,
	state: ReaderTranslateState,
	cfg: TranslatorConfig = DEFAULT_TRANSLATOR_CONFIG,
	onProgress?: (done: number, total: number) => void,
): Promise<void> {
	const doc = article.ownerDocument;

	if (state === 'off') {
		// Restore: remove translated nodes, unhide originals
		article.querySelectorAll(`.${BILINGUAL_CLASS}`).forEach(el => el.remove());
		article.querySelectorAll(`.${HIDDEN_ORIG_CLASS}`).forEach(el => {
			el.classList.remove(HIDDEN_ORIG_CLASS);
			(el as HTMLElement).style.display = '';
			// Restore textContent if we replaced it for target_only mode
			const orig = el.getAttribute(ORIG_ATTR);
			if (orig !== null && el.getAttribute(TRANS_ATTR) === el.textContent) {
				el.textContent = orig;
			}
		});
		article.querySelectorAll(`.${TRANSLATING_CLASS}`).forEach(el => el.classList.remove(TRANSLATING_CLASS));
		return;
	}

	// Collect translatable nodes
	const nodes = Array.from(article.querySelectorAll(TRANSLATABLE_SELECTOR)) as HTMLElement[];
	const items: { el: HTMLElement; text: string }[] = [];
	for (const el of nodes) {
		// Skip if no meaningful text
		const text = (el.textContent || '').trim();
		if (text.length < 2) continue;
		// Stash original on first encounter
		if (!el.hasAttribute(ORIG_ATTR)) {
			el.setAttribute(ORIG_ATTR, text);
		}
		items.push({ el, text: el.getAttribute(ORIG_ATTR) || text });
	}

	let done = 0;
	const total = items.length;
	onProgress?.(done, total);

	// Translate in batches; render incrementally so reader doesn't appear stuck
	for (let i = 0; i < items.length; i += BATCH_SIZE) {
		const slice = items.slice(i, i + BATCH_SIZE);
		// Mark as translating for visual feedback
		slice.forEach(s => s.el.classList.add(TRANSLATING_CLASS));

		const translations = await translateTexts(slice.map(s => s.text), cfg);

		for (let j = 0; j < slice.length; j++) {
			const { el } = slice[j];
			const translated = translations[j] || '';
			el.classList.remove(TRANSLATING_CLASS);
			el.setAttribute(TRANS_ATTR, translated);

			if (state === 'bilingual') {
				// Ensure original is visible (in case we came from target_only)
				if (el.classList.contains(HIDDEN_ORIG_CLASS)) {
					el.classList.remove(HIDDEN_ORIG_CLASS);
					el.style.display = '';
					const orig = el.getAttribute(ORIG_ATTR);
					if (orig) el.textContent = orig;
				}
				// Insert translated sibling right after the original
				let sibling = el.nextElementSibling;
				if (sibling && sibling.classList.contains(BILINGUAL_CLASS) && sibling.getAttribute('data-for') === el.getAttribute(ORIG_ATTR)) {
					sibling.textContent = translated;
				} else {
					// Remove any stale bilingual sibling
					if (sibling && sibling.classList.contains(BILINGUAL_CLASS)) sibling.remove();
					const trans = doc.createElement(el.tagName.toLowerCase());
					trans.className = `${BILINGUAL_CLASS} ot-${el.tagName.toLowerCase()}`;
					trans.setAttribute('data-for', el.getAttribute(ORIG_ATTR) || '');
					trans.textContent = translated;
					el.insertAdjacentElement('afterend', trans);
				}
			} else if (state === 'target_only') {
				// Replace text in place; remove any bilingual sibling
				const sibling = el.nextElementSibling;
				if (sibling && sibling.classList.contains(BILINGUAL_CLASS)) sibling.remove();
				if (translated) {
					el.textContent = translated;
				}
			}

			done++;
			onProgress?.(done, total);
		}
	}
}

export function clearTranslationCache(): void {
	cache.clear();
}

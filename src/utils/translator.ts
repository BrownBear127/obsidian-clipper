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
// Sent to model. Tolerant regex below accepts whitespace/case variants the
// model often introduces (### ||| ### / ### | ### / ###  |||  ### etc).
const SEPARATOR = '\n\n<<<SEG>>>\n\n';
const SEPARATOR_RE = /\s*<{2,3}\s*S\s*E\s*G\s*>{2,3}\s*/i;
// Backward-compat: also split on the legacy hash-pipe separator + numbered tags
const LEGACY_SEP_RE = /\s*#{2,4}\s*\|{2,4}\s*#{2,4}\s*/;
const NUMBERED_RE = /\s*\[?<?\s*SEG\s*[-#]?\d+\s*>?\]?\s*/i;
// Defuddle may wrap content in <section>/<div> so descendant selector is needed.
// Tag set chosen to avoid translating tiny inline-only <span>/<a>/<code>; we
// pick block-level text containers and de-dup nested matches below.
const TRANSLATABLE_SELECTOR = 'p, h1, h2, h3, h4, h5, h6, li, blockquote, dt, dd, figcaption';
// Also translate the standalone reader page <h1> title which lives in <main>
// outside <article>. We accept either being passed in via the `extraRoots` param.
// Skip nodes inside these — translating a <p> inside <blockquote> is double work
const SKIP_INSIDE = 'pre, code, script, style, figure, .ot-bilingual-translated';
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
		'1. 輸入有多段，段落之間以 "<<<SEG>>>" 分隔。**輸出必須保持相同段數**，每段之間以 "<<<SEG>>>" 分隔（一字不差）。',
		'2. 只輸出翻譯本文 + 分隔符，不要前後說明、不要 markdown wrapper、不要編號。',
		'3. 保留原文的標點與換行節奏。',
		'4. 專有名詞、code、URL、數字、人名、地名保持原樣不譯。',
		'5. 已是繁體中文的段落原樣輸出。',
		'6. **絕對不要把 "<<<SEG>>>" 翻譯成其他文字**，原樣保留。',
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

async function cacheKey(text: string, cfg: TranslatorConfig | string): Promise<string> {
	if (typeof cfg === 'string') return `${cfg}::${await sha256(text)}`;
	const sig = `${cfg.targetLang}::${cfg.model}::${cfg.endpoint}::${cfg.systemPrompt.length}`;
	return `${sig}::${await sha256(text)}`;
}

// Browser-polyfill aware. Falls back to global chrome.* if not available.
declare const browser: any;
declare const chrome: any;
function rt(): any {
	try { return (typeof browser !== 'undefined' && browser?.runtime) ? browser.runtime : chrome.runtime; }
	catch { return chrome.runtime; }
}

async function translateBatch(
	texts: string[],
	cfg: TranslatorConfig,
): Promise<string[]> {
	if (texts.length === 0) return [];
	const reply = await rt().sendMessage({
		action: 'ot-translate-batch',
		texts,
		config: cfg,
		separator: SEPARATOR,
	});
	if (!reply || reply.error) {
		throw new Error(reply?.error || 'translator: empty reply');
	}
	const raw: string = reply.raw || '';
	const parts = splitTolerant(raw, texts.length);
	if (parts.length === texts.length) return parts;
	throw new Error(`split mismatch: got ${parts.length}, expected ${texts.length}`);
}

/**
 * Robust translator: try batch, on split mismatch retry batch once, then
 * fall back to single-paragraph translations. Original text is kept on
 * single-failure (never silent empty).
 */
async function translateWithRetry(
	texts: string[],
	cfg: TranslatorConfig,
): Promise<string[]> {
	if (texts.length === 0) return [];
	if (texts.length === 1) {
		try {
			const parts = await translateBatch(texts, cfg);
			return parts;
		} catch (err) {
			console.warn('[Translator] single failed, keep original:', err);
			return [texts[0]]; // keep original on hard failure
		}
	}
	try { return await translateBatch(texts, cfg); }
	catch (err1) {
		console.warn('[Translator] batch retry due to:', err1);
		try { return await translateBatch(texts, cfg); }
		catch (err2) {
			console.warn('[Translator] batch failed twice, downgrading to per-paragraph:', err2);
			const out: string[] = [];
			for (const t of texts) {
				const single = await translateWithRetry([t], cfg);
				out.push(single[0]);
			}
			return out;
		}
	}
}

/**
 * Split model output into N parts, tolerating separator drift.
 * Tries the canonical separator first, then legacy/looser variants.
 */
function splitTolerant(raw: string, expectedCount: number): string[] {
	const candidates: RegExp[] = [SEPARATOR_RE, LEGACY_SEP_RE, NUMBERED_RE];
	for (const re of candidates) {
		const parts = raw.split(re).map(p => p.trim()).filter(p => p.length > 0);
		if (parts.length === expectedCount) return parts;
		// Within ±1 also acceptable (model may emit trailing empty)
		if (Math.abs(parts.length - expectedCount) <= 1 && parts.length >= expectedCount) return parts.slice(0, expectedCount);
	}
	// Fallback: blank line split
	const byBlank = raw.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
	if (byBlank.length === expectedCount) return byBlank;
	// Last resort: try the canonical regex even if mismatched count, let caller pad/truncate
	return raw.split(SEPARATOR_RE).map(p => p.trim());
}

export async function translateTexts(
	texts: string[],
	cfg: TranslatorConfig = DEFAULT_TRANSLATOR_CONFIG,
): Promise<string[]> {
	const out: string[] = new Array(texts.length).fill('');
	const need: { idx: number; text: string }[] = [];

	// Cache check
	for (let i = 0; i < texts.length; i++) {
		const key = await cacheKey(texts[i], cfg);
		const hit = cache.get(key);
		if (hit !== undefined) {
			out[i] = hit;
		} else {
			need.push({ idx: i, text: texts[i] });
		}
	}

	// Batch the misses with retry+fallback
	for (let i = 0; i < need.length; i += BATCH_SIZE) {
		const slice = need.slice(i, i + BATCH_SIZE);
		const translations = await translateWithRetry(slice.map(s => s.text), cfg);
		for (let j = 0; j < slice.length; j++) {
			const t = translations[j] || slice[j].text; // last-resort: keep original
			out[slice[j].idx] = t;
			const key = await cacheKey(slice[j].text, cfg);
			// Only cache when we actually translated (not when we fell back to original)
			if (t && t !== slice[j].text) cache.set(key, t);
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
	extraRoots: HTMLElement[] = [],
): Promise<void> {
	const doc = article.ownerDocument;
	// extraRoots (e.g. main > h1 title) are translated in addition to article children

	if (state === 'off') {
		// Remove inserted bilingual siblings
		article.querySelectorAll(`.${BILINGUAL_CLASS}`).forEach(el => el.remove());
		// Restore any node whose textContent we replaced (target_only path)
		article.querySelectorAll(`[${ORIG_ATTR}]`).forEach(el => {
			const orig = el.getAttribute(ORIG_ATTR);
			const html = el.getAttribute(`${ORIG_ATTR}-html`);
			if (html !== null) {
				// Restore inner HTML if we stashed it
				(el as HTMLElement).innerHTML = html;
			} else if (orig !== null && el.textContent !== orig) {
				el.textContent = orig;
			}
			el.classList.remove(HIDDEN_ORIG_CLASS, TRANSLATING_CLASS);
			(el as HTMLElement).style.display = '';
		});
		return;
	}

	// Collect translatable nodes (descendant search + nested de-dup)
	const nodes = [
		...Array.from(article.querySelectorAll(TRANSLATABLE_SELECTOR)),
		...extraRoots,
	] as HTMLElement[];
	const items: { el: HTMLElement; text: string }[] = [];
	for (const el of nodes) {
		// Skip if inside a node we shouldn't translate (pre/code/figure/already translated)
		if (el.closest(SKIP_INSIDE)) continue;
		// Skip our own translation siblings
		if (el.classList.contains(BILINGUAL_CLASS)) continue;
		// Skip if a translatable ancestor already covers this (e.g. <li><p>)
		let parent = el.parentElement;
		let nestedInside = false;
		while (parent && parent !== article) {
			if (parent.matches(TRANSLATABLE_SELECTOR)) { nestedInside = true; break; }
			parent = parent.parentElement;
		}
		if (nestedInside) continue;
		// Skip if no meaningful text
		const text = (el.textContent || '').trim();
		if (text.length < 2) continue;
		// Stash original on first encounter — both text (for translator input)
		// and innerHTML (so we can fully restore inline links/bold/etc on 'off')
		if (!el.hasAttribute(ORIG_ATTR)) {
			el.setAttribute(ORIG_ATTR, text);
			el.setAttribute(`${ORIG_ATTR}-html`, el.innerHTML);
		}
		items.push({ el, text: el.getAttribute(ORIG_ATTR) || text });
	}
	console.log(`[Translator] collected ${items.length} translatable nodes from ${nodes.length} candidates`);

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
				// Restore original innerHTML (in case we came from target_only)
				const stashedHtml = el.getAttribute(`${ORIG_ATTR}-html`);
				if (stashedHtml !== null && el.innerHTML !== stashedHtml) {
					el.innerHTML = stashedHtml;
				}
				// Insert/refresh translated sibling right after the original
				let sibling = el.nextElementSibling;
				if (sibling && sibling.classList.contains(BILINGUAL_CLASS) && sibling.getAttribute('data-for') === el.getAttribute(ORIG_ATTR)) {
					sibling.textContent = translated;
				} else {
					if (sibling && sibling.classList.contains(BILINGUAL_CLASS)) sibling.remove();
					const trans = doc.createElement(el.tagName.toLowerCase());
					trans.className = `${BILINGUAL_CLASS} ot-${el.tagName.toLowerCase()}`;
					trans.setAttribute('data-for', el.getAttribute(ORIG_ATTR) || '');
					trans.textContent = translated;
					el.insertAdjacentElement('afterend', trans);
				}
			} else if (state === 'target_only') {
				// Remove any bilingual sibling first
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

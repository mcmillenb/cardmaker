import 'dotenv/config';
import express from 'express';
import sharp from 'sharp';
import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
app.use(express.json({ limit: '20mb' }));
app.use(express.static('public'));

// ---- Saved generations ----
// Every render is written to data/cards/{scryfallId}/ (art.png = the current art, which
// may have a face swapped in; original.png = the pre-swap art, when there is one;
// meta.json = card details, prompt and notes), so results survive page reloads and
// server restarts. Deck lists are cached in data/decks/{id}.json. Art is served
// statically from /data/cards so a restored 100-card deck doesn't arrive as one huge
// JSON response of base64 images.
const DATA_DIR = process.env.CARDMAKER_DATA_DIR || path.join(__dirname, 'data');
const CARDS_DIR = path.join(DATA_DIR, 'cards');
const DECKS_DIR = path.join(DATA_DIR, 'decks');
const SCRYFALL_ID_RE = /^[0-9a-f-]{36}$/i;
app.use('/data/cards', express.static(CARDS_DIR));

const cardDir = (id) => path.join(CARDS_DIR, id.toLowerCase());

async function readJson(file) {
  try {
    return JSON.parse(await fs.readFile(file, 'utf8'));
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw err;
  }
}

async function writeJsonAtomic(file, value) {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(value, null, 2));
  await fs.rename(tmp, file);
}

// The shape the frontend gets back for a saved card: same fields as a fresh render,
// with image URLs (cache-busted by save time) instead of data URLs.
async function loadSavedCard(id) {
  const meta = await readJson(path.join(cardDir(id), 'meta.json'));
  if (!meta) return null;
  const base = `/data/cards/${id.toLowerCase()}`;
  return {
    ...meta,
    image: `${base}/art.png?v=${meta.savedAt}`,
    originalImage: meta.hasOriginal ? `${base}/original.png?v=${meta.savedAt}` : undefined,
    saved: true,
  };
}

// image / originalImage may each be a PNG data URL (new pixels), one of this card's
// own saved-file URLs (e.g. revert copies original.png back over art.png), or, for
// originalImage only, null/undefined to drop it.
async function resolveImageInput(id, value) {
  if (!value) return null;
  if (value.startsWith('data:')) return decodeDataUrlOrBase64(value);
  const match = /^\/data\/cards\/([0-9a-f-]{36})\/(art|original)\.png(\?.*)?$/i.exec(value);
  if (!match || match[1].toLowerCase() !== id.toLowerCase()) {
    throw new Error('Images must be data URLs or this card\'s own saved files.');
  }
  return fs.readFile(path.join(cardDir(id), `${match[2]}.png`));
}

async function saveCard(id, { card, prompt, faceCheck, faceNote }, image, originalImage) {
  const dir = cardDir(id);
  await fs.mkdir(dir, { recursive: true });
  // Both buffers are resolved before anything is written, since either may be read
  // from the files about to be overwritten.
  await fs.writeFile(path.join(dir, 'art.png'), image);
  if (originalImage) {
    await fs.writeFile(path.join(dir, 'original.png'), originalImage);
  } else {
    await fs.rm(path.join(dir, 'original.png'), { force: true });
  }
  await writeJsonAtomic(path.join(dir, 'meta.json'), {
    card,
    prompt,
    faceCheck,
    faceNote,
    hasOriginal: !!originalImage,
    savedAt: Date.now(),
  });
  return loadSavedCard(id);
}

const IMAGE_MODEL = '@cf/stabilityai/stable-diffusion-xl-base-1.0';

// Matches the actual art-window opening in the card frame we composite onto (see
// composeCard/LAYOUT.artWindow in public/index.html), not the full card — most of a
// full 5:7 card is covered by the frame (title bar, borders, text box) once overlaid,
// so generating full-card pixels there would just be paying Cloudflare to draw
// artwork that never shows. 632x464 keeps the art window's ~1.37 aspect ratio at
// multiples of 8 for SDXL.
const ART_WIDTH = 632;
const ART_HEIGHT = 464;

function parseScryfallUrl(rawUrl) {
  const url = new URL(rawUrl);

  if (url.hostname === 'api.scryfall.com') {
    return url.toString();
  }

  if (url.hostname === 'scryfall.com') {
    const segments = url.pathname.split('/').filter(Boolean);
    // /card/{set}/{number}/{name-slug}  or  /card/{set}/{number}/{lang}/{name-slug}
    if (segments[0] === 'card' && segments[1] && segments[2]) {
      const set = segments[1];
      const number = segments[2];
      return `https://api.scryfall.com/cards/${encodeURIComponent(set)}/${encodeURIComponent(number)}`;
    }
  }

  throw new Error('That does not look like a Scryfall card URL.');
}

const COLOR_NAMES = { W: 'white', U: 'blue', B: 'black', R: 'red', G: 'green' };

// Spelled out rather than passing along the raw "{2}{B}{B}" shorthand, so the image
// model actually parses it as a color instruction instead of ignoring/misreading symbols.
// Returns null for colorless cards (artifacts, lands) rather than an instruction to render
// them grey/neutral — colorless just means "no card color to bias toward," not "no color in
// the art." The subject and flavor text should drive the palette instead.
function describeColors(card) {
  const names = (card.colors || []).map((c) => COLOR_NAMES[c] || c);
  if (names.length === 0) {
    return null;
  }
  const joined =
    names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
  const verb = names.length > 1 ? 'are' : 'is';
  return `This card's color${names.length > 1 ? 's' : ''} ${verb} ${joined}. Those colors should feature prominently in the color palette of the generated image.`;
}

// Face focus only applies to card types that usually depict a character.
function wantsFaceFocus(card, faceFocus) {
  return !!faceFocus && /Creature|Planeswalker|Legendary/.test(card.type_line || '');
}

function buildArtPrompt(card, options = {}) {
  const parts = [];
  const faceFocus = wantsFaceFocus(card, options.faceFocus);
  // SDXL's text encoder only reads roughly the first 77 tokens, so framing has to lead
  // the prompt — anything after the rules text is effectively ignored. Used only when
  // the user has face photos loaded: the default full-figure composition leaves heads
  // ~50px tall in the 632x464 art, too small for a swapped face to read (or sometimes
  // to be detected at all).
  // The person's own description (hair, beard, age...) goes first of all: the face swap
  // only replaces the inner face, so hair, face shape and skin tone have to come from
  // the generated character for the result to read as that person.
  if (faceFocus) {
    const subject = options.personDescription ? `one character: ${options.personDescription},` : 'one character';
    parts.push(`Chest-up painterly fantasy oil painting of ${subject} in their setting, face clearly visible, front-facing, well lit and unobstructed.`);
  }
  parts.push(`Fantasy illustration for a Magic: The Gathering card named "${card.name}".`);
  if (card.type_line) parts.push(`Card type: ${card.type_line}.`);
  const colorNote = describeColors(card);
  if (colorNote) parts.push(colorNote);
  if (card.oracle_text) parts.push(`Rules text: ${card.oracle_text.replace(/\n/g, ' ')}`);
  if (card.flavor_text) parts.push(`Flavor text: "${card.flavor_text.replace(/\n/g, ' ')}"`);
  if (card.power && card.toughness) parts.push(`It is a creature with power ${card.power} and toughness ${card.toughness}.`);
  const composition = faceFocus
    ? 'chest-up framing'
    : 'vertical portrait composition with the subject centered and room above and below';
  parts.push(`Style: dramatic painterly fantasy digital art, single focal subject, richly detailed, cinematic lighting, ${composition}. Do not include any text, letters, numbers, borders, or card frame in the image — artwork only.`);
  return parts.join(' ');
}

// Free-text description of the person, typed (or pre-filled) in the UI. Kept short so
// it can't crowd the card's own details out of SDXL's ~77-token window.
function cleanPersonDescription(raw) {
  if (typeof raw !== 'string') return '';
  return raw.replace(/["\n\r]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 120);
}

const NEGATIVE_PROMPT =
  'text, letters, words, numbers, watermark, signature, logo, mana symbol, card frame, border, UI, template, trading card layout';

// Added when faceFocus applies, so the swap has an unobstructed face to work with.
const FACE_FOCUS_NEGATIVE = 'helmet covering face, visor, mask, face in shadow, back turned, profile view, tiny distant figure, full body, multiple panels, collage, card, frame within frame';

function decodeDataUrlOrBase64(input) {
  const match = /^data:.*;base64,(.*)$/s.exec(input);
  return Buffer.from(match ? match[1] : input, 'base64');
}

async function callWorkersAI(model, body) {
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
  const token = process.env.CLOUDFLARE_API_TOKEN;
  if (!accountId || !token) {
    throw new Error('Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN in your .env file.');
  }

  const resp = await fetch(
    `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${model}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }
  );

  const contentType = resp.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    const data = await resp.json();
    if (!resp.ok || data.success === false) {
      const message = data.errors?.[0]?.message || `Cloudflare Workers AI request failed (${resp.status}).`;
      throw new Error(message);
    }
    const base64 = data.result?.image;
    if (!base64) throw new Error('Cloudflare Workers AI returned no image data.');
    return Buffer.from(base64, 'base64');
  }

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `Cloudflare Workers AI request failed (${resp.status}).`);
  }
  return Buffer.from(await resp.arrayBuffer());
}

// Confirmed by testing: this model intermittently returns a solid-black image for the
// exact same request (2 of 3 identical retries came back black) — a known SD1.5 flaky
// failure mode, unrelated to prompt content or input. Detect and retry rather than
// surfacing a black result to the user.
async function isNearBlack(buffer) {
  const stats = await sharp(buffer).stats();
  return Math.max(...stats.channels.map((c) => c.mean)) < 3;
}

async function callWorkersAIWithBlankRetry(model, body, attempts = 4) {
  let lastBuffer;
  for (let i = 0; i < attempts; i++) {
    lastBuffer = await callWorkersAI(model, body);
    if (!(await isNearBlack(lastBuffer))) return lastBuffer;
  }
  throw new Error('The image model kept returning a blank result (a known flaky failure on Cloudflare\'s side) — please try again.');
}

// A fast, free stand-in for generateBaseArt, used when the "skip AI art" testing
// toggle is on — lets the card-construction/typesetting side (frame, fonts, sizing)
// be iterated on without waiting on or paying for a Cloudflare generation each time.
async function generatePlaceholderArt() {
  const svg = `<svg width="${ART_WIDTH}" height="${ART_HEIGHT}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#53535c"/>
    <text x="50%" y="50%" font-family="sans-serif" font-size="26" fill="#d8d8de" text-anchor="middle" dominant-baseline="middle">art generation skipped</text>
  </svg>`;
  return sharp(Buffer.from(svg)).png().toBuffer();
}

async function generateBaseArt(prompt, negativePrompt = NEGATIVE_PROMPT) {
  const rawBuffer = await callWorkersAIWithBlankRetry(IMAGE_MODEL, {
    prompt,
    negative_prompt: negativePrompt,
    width: ART_WIDTH,
    height: ART_HEIGHT,
    num_steps: 16,
  });

  // SDXL generates natively at ART_WIDTH x ART_HEIGHT, so this just guarantees the
  // exact pixel size without cropping anything out — a no-op unless the model rounds.
  return sharp(rawBuffer).resize(ART_WIDTH, ART_HEIGHT, { fit: 'fill' }).png().toBuffer();
}

// With faceFocus, a render is only useful if its art has a face big enough and
// front-facing enough for the swap to read as the person. The detector check is local
// and fast, so regenerate a few times rather than hand back a helmet, a profile or a
// tiny distant figure. Each retry is one more Cloudflare generation.
const FACE_ATTEMPTS = 3;
const MIN_FACE_HEIGHT = 80; // px, in the 632x464 art
const MAX_FACE_YAW = 0.3; // see frontalness() in faceswap_server.py

function isUsableFace(face) {
  return !!face && face.height >= MIN_FACE_HEIGHT && face.yaw <= MAX_FACE_YAW;
}

// Ranks fallbacks when no attempt is fully usable: any face beats none, and a
// front-facing face beats a bigger turned one.
function faceRank(face) {
  if (!face) return -1;
  return face.height + (face.yaw <= MAX_FACE_YAW ? 1000 : 0);
}

async function generateArtWithFace(prompt, negativePrompt) {
  let best = null;
  for (let attempt = 1; attempt <= FACE_ATTEMPTS; attempt++) {
    let buffer;
    try {
      buffer = await generateBaseArt(prompt, negativePrompt);
    } catch (err) {
      if (best) {
        best.attempts = attempt;
        break; // keep what we have rather than fail the whole card
      }
      throw err;
    }
    let face;
    try {
      face = (await callFaceswap('/inspect', { image: buffer.toString('base64') })).face;
    } catch {
      // Face server unavailable: no way to judge the art, so retrying is pointless.
      return { buffer, attempts: attempt, usable: false };
    }
    if (isUsableFace(face)) return { buffer, attempts: attempt, usable: true };
    if (!best || faceRank(face) > faceRank(best.face)) best = { buffer, face };
    best.attempts = attempt;
  }
  return { buffer: best.buffer, attempts: best.attempts, usable: false };
}

// Real identity-preserving face swap (inswapper via insightface), run locally and for free
// in a persistent Python process — see faceswap_server.py. This replaced an earlier
// Cloudflare-inpainting-based approximation that was both unreliable (flaky black/garbled
// output) and not actually identity-preserving; this local model is neither.
const FACESWAP_PORT = process.env.FACESWAP_PORT || 8787;
const FACESWAP_URL = `http://127.0.0.1:${FACESWAP_PORT}`;
let faceswapProcess;
let faceswapReady;

function startFaceswapServer() {
  const pythonBin = path.join(__dirname, 'faceswap-env', 'bin', 'python');
  faceswapProcess = spawn(pythonBin, [path.join(__dirname, 'faceswap_server.py')], {
    env: { ...process.env, FACESWAP_PORT: String(FACESWAP_PORT) },
  });
  faceswapProcess.stdout.on('data', (d) => process.stdout.write(`[faceswap] ${d}`));
  faceswapProcess.stderr.on('data', (d) => process.stderr.write(`[faceswap] ${d}`));
  faceswapProcess.on('exit', (code) => {
    console.log(`[faceswap] process exited (${code})`);
  });

  faceswapReady = (async () => {
    for (let i = 0; i < 60; i++) {
      try {
        const resp = await fetch(`${FACESWAP_URL}/health`);
        if (resp.ok) {
          console.log('[faceswap] ready');
          return;
        }
      } catch {
        // not up yet
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    throw new Error('faceswap_server.py did not become ready in time.');
  })();
  // Face swap is optional: don't let a failed startup crash the server as an unhandled
  // rejection. mergeFace still rejects when it awaits faceswapReady.
  faceswapReady.catch((err) => console.error(`[faceswap] ${err.message}`));
}

async function callFaceswap(route, body) {
  await faceswapReady;
  const resp = await fetch(`${FACESWAP_URL}${route}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) {
    const err = new Error(data.error || 'Face swap failed.');
    err.status = resp.status;
    err.code = data.code;
    err.photos = data.photos;
    throw err;
  }
  return data;
}

// `identity` is either a single face photo (Buffer) or a precomputed embedding (array
// of 512 numbers from buildFaceIdentity, averaged over several photos of one person).
async function mergeFace(cardBuffer, identity, region) {
  const data = await callFaceswap('/swap', {
    ...(Buffer.isBuffer(identity) ? { source: identity.toString('base64') } : { embedding: identity }),
    target: cardBuffer.toString('base64'),
    region: region ? { cx: region.cx, cy: region.cy, rx: region.rx, ry: region.ry } : undefined,
  });
  return { image: Buffer.from(data.image, 'base64'), approximate: !!data.approximate };
}

async function buildFaceIdentity(photoBuffers) {
  return callFaceswap('/identity', { photos: photoBuffers.map((b) => b.toString('base64')) });
}

// Double-faced / adventure / split cards have no top-level oracle_text or colors on Scryfall;
// fall back to the front face so prompts aren't built from empty fields.
function normalizeCard(card) {
  const face = card.card_faces?.[0];
  if (!face) return card;
  return {
    ...card,
    mana_cost: card.mana_cost || face.mana_cost,
    oracle_text: card.oracle_text || face.oracle_text,
    flavor_text: card.flavor_text || face.flavor_text,
    power: card.power ?? face.power,
    toughness: card.toughness ?? face.toughness,
    colors: card.colors || face.colors,
  };
}

function parseArchidektDeckId(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch {
    throw new Error('That does not look like an Archidekt deck URL.');
  }
  const match = /^\/decks\/(\d+)/.exec(url.pathname);
  if (!/(^|\.)archidekt\.com$/.test(url.hostname) || !match) {
    throw new Error('That does not look like an Archidekt deck URL (expected archidekt.com/decks/{id}).');
  }
  return match[1];
}

app.post('/api/deck', async (req, res) => {
  const { url } = req.body || {};
  if (!url) return res.status(400).json({ error: 'Provide an Archidekt deck URL.' });

  let deckId;
  try {
    deckId = parseArchidektDeckId(url);
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }

  let deck;
  try {
    const resp = await fetch(`https://archidekt.com/api/decks/${deckId}/`, {
      headers: { 'User-Agent': 'cardmaker/1.0', Accept: 'application/json' },
    });
    if (resp.status === 404 || resp.status === 403) {
      throw new Error('Deck not found (it may be private).');
    }
    if (!resp.ok) throw new Error(`Archidekt lookup failed (${resp.status}).`);
    deck = await resp.json();
  } catch (err) {
    // Fall back to the last copy of this deck we fetched, so a saved deck can still be
    // reopened when Archidekt is down or rate-limiting.
    const cached = await readJson(path.join(DECKS_DIR, `${deckId}.json`)).catch(() => null);
    if (cached) return res.json({ ...cached, cached: true });
    return res.status(502).json({ error: `Could not fetch deck from Archidekt: ${err.message}` });
  }

  // Skip Maybeboard/Sideboard-style categories that Archidekt marks as not part of the deck.
  const excluded = new Set(
    (deck.categories || []).filter((c) => c.includedInDeck === false).map((c) => c.name)
  );

  const cards = [];
  for (const entry of deck.cards || []) {
    const cats = entry.categories || [];
    if (cats.length > 0 && cats.every((c) => excluded.has(c))) continue;
    const scryfallId = entry.card?.uid;
    if (!scryfallId) continue;
    cards.push({
      scryfallId,
      name: entry.card.oracleCard?.name || entry.card.displayName || 'Unknown card',
      quantity: entry.quantity || 1,
      category: cats[0] || null,
    });
  }

  const result = { id: deckId, name: deck.name, cards };
  await writeJsonAtomic(path.join(DECKS_DIR, `${deckId}.json`), result).catch((err) =>
    console.error(`Could not cache deck ${deckId}: ${err.message}`)
  );
  res.json(result);
});

// Saved generations for many cards at once (deck restore). Missing ids are omitted.
app.post('/api/cards/lookup', async (req, res) => {
  const ids = Array.isArray(req.body?.ids) ? req.body.ids.filter((id) => SCRYFALL_ID_RE.test(id)) : [];
  const cards = {};
  await Promise.all(ids.map(async (id) => {
    const saved = await loadSavedCard(id).catch(() => null);
    if (saved) cards[id] = saved;
  }));
  res.json({ cards });
});

app.get('/api/cards/:id', async (req, res) => {
  if (!SCRYFALL_ID_RE.test(req.params.id)) return res.status(400).json({ error: 'Invalid card id.' });
  const saved = await loadSavedCard(req.params.id);
  if (!saved) return res.status(404).json({ error: 'No saved art for this card.' });
  res.json(saved);
});

// Updates a saved card after client-side changes (face swap, manual swap, revert).
app.put('/api/cards/:id', async (req, res) => {
  const { id } = req.params;
  if (!SCRYFALL_ID_RE.test(id)) return res.status(400).json({ error: 'Invalid card id.' });
  const existing = await readJson(path.join(cardDir(id), 'meta.json'));
  if (!existing) return res.status(404).json({ error: 'No saved art for this card.' });
  const { image, originalImage, faceNote } = req.body || {};
  if (!image) return res.status(400).json({ error: 'Provide image.' });
  let imageBuf;
  let originalBuf;
  try {
    imageBuf = await resolveImageInput(id, image);
    originalBuf = await resolveImageInput(id, originalImage);
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }
  res.json(await saveCard(id, { ...existing, faceNote: faceNote ?? existing.faceNote }, imageBuf, originalBuf));
});

app.post('/api/render', async (req, res) => {
  const { url, scryfallId, skipArt, faceFocus, useSaved } = req.body || {};
  const personDescription = cleanPersonDescription(req.body?.personDescription);
  if (!url && !scryfallId) {
    return res.status(400).json({ error: 'Provide a Scryfall card URL.' });
  }

  let apiUrl;
  if (scryfallId) {
    if (!/^[0-9a-f-]{36}$/i.test(scryfallId)) {
      return res.status(400).json({ error: 'Invalid scryfallId.' });
    }
    apiUrl = `https://api.scryfall.com/cards/${scryfallId}`;
  } else {
    try {
      apiUrl = parseScryfallUrl(url);
    } catch (err) {
      return res.status(400).json({ error: err.message });
    }
  }

  let card;
  try {
    const scryfallResp = await fetch(apiUrl, {
      headers: { 'User-Agent': 'cardmaker/1.0', Accept: 'application/json' },
    });
    if (!scryfallResp.ok) {
      throw new Error(`Scryfall lookup failed (${scryfallResp.status}).`);
    }
    card = normalizeCard(await scryfallResp.json());
  } catch (err) {
    return res.status(502).json({ error: `Could not fetch card from Scryfall: ${err.message}` });
  }

  // Opening a card that already has saved art shows that instead of spending another
  // generation; regenerating is an explicit request (useSaved false).
  if (useSaved) {
    const saved = await loadSavedCard(card.id).catch(() => null);
    if (saved) return res.json({ ...saved, fromSaved: true });
  }

  const prompt = buildArtPrompt(card, { faceFocus: !!faceFocus, personDescription });
  const focused = wantsFaceFocus(card, faceFocus);
  const negativePrompt = focused ? `${NEGATIVE_PROMPT}, ${FACE_FOCUS_NEGATIVE}` : NEGATIVE_PROMPT;

  let cardBuffer;
  let faceCheck;
  try {
    if (skipArt) {
      cardBuffer = await generatePlaceholderArt();
    } else if (focused) {
      const { buffer, ...check } = await generateArtWithFace(prompt, negativePrompt);
      cardBuffer = buffer;
      faceCheck = check;
    } else {
      cardBuffer = await generateBaseArt(prompt, negativePrompt);
    }
  } catch (err) {
    return res.status(502).json({ error: `Image generation failed: ${err.message}` });
  }

  const cardInfo = {
    id: card.id,
    name: card.name,
    mana_cost: card.mana_cost,
    type_line: card.type_line,
    oracle_text: card.oracle_text,
    flavor_text: card.flavor_text,
    power: card.power,
    toughness: card.toughness,
    colors: card.colors || [],
    set_name: card.set_name,
    rarity: card.rarity,
  };

  // Placeholder art is for testing the frame only; don't let it overwrite real saved art.
  if (skipArt) {
    return res.json({ card: cardInfo, prompt, image: `data:image/png;base64,${cardBuffer.toString('base64')}` });
  }
  try {
    res.json(await saveCard(card.id, { card: cardInfo, prompt, faceCheck }, cardBuffer, null));
  } catch (err) {
    // Saving is a convenience; still hand back the art that was just paid for.
    console.error(`Could not save card ${card.id}: ${err.message}`);
    res.json({ card: cardInfo, prompt, faceCheck, image: `data:image/png;base64,${cardBuffer.toString('base64')}` });
  }
});

app.post('/api/face-identity', async (req, res) => {
  const { photos } = req.body || {};
  if (!Array.isArray(photos) || photos.length === 0) {
    return res.status(400).json({ error: 'Provide one or more face photos.' });
  }

  let buffers;
  try {
    buffers = photos.map(decodeDataUrlOrBase64);
  } catch {
    return res.status(400).json({ error: 'Could not decode the photos.' });
  }

  try {
    res.json(await buildFaceIdentity(buffers));
  } catch (err) {
    res.status(err.status === 400 ? 400 : 502).json({ error: err.message, photos: err.photos });
  }
});

app.post('/api/merge-face', async (req, res) => {
  const { image, faceImage, embedding, region } = req.body || {};
  if (!image || (!faceImage && !embedding)) {
    return res.status(400).json({ error: 'Provide image and either faceImage or embedding.' });
  }
  if (embedding && (!Array.isArray(embedding) || embedding.length !== 512 || !embedding.every((n) => typeof n === 'number'))) {
    return res.status(400).json({ error: 'embedding must be an array of 512 numbers.' });
  }
  if (region && (typeof region.cx !== 'number' || typeof region.cy !== 'number')) {
    return res.status(400).json({ error: 'region, if provided, must have numeric cx and cy.' });
  }

  let cardBuffer;
  let identity = embedding;
  try {
    cardBuffer = decodeDataUrlOrBase64(image);
    if (!identity) identity = decodeDataUrlOrBase64(faceImage);
  } catch (err) {
    return res.status(400).json({ error: 'Could not decode image or faceImage.' });
  }

  let merged;
  try {
    merged = await mergeFace(cardBuffer, identity, region);
  } catch (err) {
    // 422 = the art simply has no detectable face (common for lands, artifacts,
    // non-humanoid creatures) — a normal outcome, not a server failure.
    const status = err.code === 'no-target-face' ? 422 : 502;
    return res.status(status).json({ error: `Face swap failed: ${err.message}`, code: err.code });
  }

  res.json({ image: `data:image/png;base64,${merged.image.toString('base64')}`, approximate: merged.approximate });
});

startFaceswapServer();

const port = process.env.PORT || 3000;
const server = app.listen(port, () => {
  console.log(`cardmaker running at http://localhost:${port}`);
});

function shutdown() {
  if (faceswapProcess) faceswapProcess.kill();
  server.close(() => process.exit(0));
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

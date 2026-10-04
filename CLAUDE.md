# cardmaker

Generates AI artwork for Magic: The Gathering cards from their Scryfall text details (never from the real illustration). Supports a single Scryfall card link or a whole Archidekt deck, with an optional local face swap.

## Running

- `npm start` runs `server.js` (Express, ES modules) on `PORT` (default 3000) and serves `public/`.
- Copy `.env.example` to `.env` and set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (Workers AI). Optional: `PORT`, `FACESWAP_PORT` (default 8787).
- On startup the server spawns `faceswap_server.py` using `faceswap-env/bin/python` and waits for its `/health` check. `faceswap-env/` and `faceswap-models/` are local and large; don't edit or commit them.

## Architecture

- `server.js` — all backend logic:
  - `parseScryfallUrl` turns a scryfall.com card URL into an api.scryfall.com URL.
  - `normalizeCard` copies front-face fields (oracle text, colors, mana cost, P/T, flavor) to the top level for double-faced, adventure and split cards, which lack them on Scryfall.
  - `buildArtPrompt` and `describeColors` build the image prompt. Colors are spelled out in words rather than passed as `{B}` symbols.
  - `generateBaseArt` calls Cloudflare SDXL at `ART_WIDTH`x`ART_HEIGHT` (632x464 — the
    card frame's art-window opening, not the full 5:7 card; see `LAYOUT.artWindow` in
    `public/index.html`, since most of a full card gets covered by the frame anyway)
    with a retry on the model's known near-black failure (`callWorkersAIWithBlankRetry`).
  - `mergeFace` proxies to the local Python face-swap server.
- `faceswap_server.py` — Flask app using insightface (`buffalo_l` detector, `inswapper_128_fp16` swapper) on CPU. Loads models once at startup. `/identity` blends several photos of one person into a single averaged ArcFace embedding (dropping photos whose face doesn't match the others, by leave-one-out cosine similarity); `/swap` accepts that embedding or a single photo. The detector runs at a low threshold (`DET_THRESH`) and callers filter by score. For a user-marked region, `find_marked_face` tries the full image, then a zoomed-in crop of the mark, then fits ArcFace's canonical 5-point layout to the marked oval (returned as `approximate: true`). Swaps run `SWAP_PASSES` (2) times on the same landmarks, which pulls the result further toward the person's identity. `/inspect` reports the largest face's height and `frontalness` (nose offset from the eye midpoint / eye distance) for the render retry loop. `/identity` also returns a median age and majority sex, used only to pre-fill the description. Face swapping never calls Cloudflare.
- `public/index.html` — single-file frontend (inline CSS and JS, no build step).
  - `composeCard` overlays the generated art into a real Magic card frame (title, mana
    cost, type line, rules/flavor text, P/T box) entirely with `<canvas>` — no
    server/Cloudflare involvement, so it costs nothing beyond the art generation that
    already happened. Frame images live in `public/frames/{category}-frame.png` (per
    color identity: white/blue/black/red/green/gold/colorless/artifact/colorlessLand)
    plus `-legendary.png` (an additive title-bar crest, not a frame replacement — see
    `data/borders/m15/border.js` in the source repo for why) and `-pt.png`. Mana/ability
    symbols are `public/mana/{0-51}.png`, indexed by the `MANA_SYMBOL_CODES` array.
    Layout pixel coordinates (`LAYOUT` in index.html) match a 749x1044 canvas, taken
    directly from that same `border.js`.
  - Title/type/P&T use the real Beleren font and rules/flavor text the real MPlantin
    font (`public/fonts/*.ttf`, loaded via `@font-face`) — the same faces real cards
    use, not a lookalike substitute.
  - These assets (frame art, mana symbols, fonts) are vendored from the open-source
    Card Conjurer project (github.com/shopglobal/cardconjurer, `data/borders`,
    `data/manaSymbols`, `data/fonts`). Per that repo's `TermsOfUse.txt`, they're for
    personal/non-commercial use only.
  - The vendored MPlantin/MPlantin-italic/Beleren font files all define the Unicode
    minus sign (U+2212, which Scryfall uses in planeswalker loyalty costs like "−3:")
    as an empty/contourless glyph — it silently draws as nothing. `smartenText` swaps
    it for a plain hyphen before rendering.
  - Rules-text layout distinguishes wrapped lines within one ability (tight leading)
    from separate abilities/paragraphs (extra gap on top) — see the `layoutRulesText`/
    `fitAndDrawRulesText` comments, which follow Card Conjurer's own `drawText` in its
    `index.html` (a plain wrap advances by `textSize + 1`; an explicit paragraph break
    adds its `lineSpace` on top of that).
  - When a card has both rules and flavor text, `fitAndDrawRulesAndFlavor` puts the rules at
    the top of the box and the flavor against its bottom, shrinking both together until they
    fit with a minimum gap; any spare room becomes the gap between them. Bottom limits
    (`textBoxBottom`, `ptY + 2` for creatures) are the lowest Y the text ink may reach, not
    the last line's top.

## API

- `POST /api/deck` `{ url }` — Archidekt deck URL (`archidekt.com/decks/{id}`). Fetches `https://archidekt.com/api/decks/{id}/` (public decks only) and returns `{ id, name, cards: [{ scryfallId, name, quantity, category }] }`. `scryfallId` is Archidekt's `card.uid`, which is the Scryfall card ID. Cards whose categories are all marked `includedInDeck: false` (maybeboard, sideboard) are skipped. This endpoint does not generate images.
- `POST /api/render` `{ url }` or `{ scryfallId }` — looks up the card on Scryfall, generates art, saves it (see Saved data), and returns `{ card (incl. id), prompt, image, saved: true, ... }` with `image` as a `/data/cards/...` URL. With `useSaved: true`, a card that already has saved art returns that instead (`fromSaved: true`) without generating. With `skipArt`, placeholder art is returned as a data URL and never saved. A generation takes roughly 10-60 seconds.
- `GET /api/cards/:id`, `POST /api/cards/lookup { ids }` → `{ cards: { id: saved } }` — read saved cards.
- `PUT /api/cards/:id { image, originalImage, faceNote }` — save client-side changes (face swaps, revert). Each image is a data URL or one of that card's own `/data/cards/{id}/(art|original).png` URLs; `originalImage: null` removes it.
- `POST /api/render` also accepts `faceFocus: true`, which (for creature/planeswalker/legendary type lines, see `wantsFaceFocus`) puts a chest-up, face-visible framing sentence at the *start* of the prompt and adds face-hiding terms to the negative prompt. It has to lead: SDXL's text encoder reads only ~77 tokens, so anything after the rules text is ignored. Without it, the art has full-body figures whose faces are ~50px, too small for a swap to show. With it, `personDescription` (free text such as hair, beard, age; cleaned and capped at 120 chars by `cleanPersonDescription`) goes into that leading sentence, because inswapper only replaces the inner face and hair, face shape and skin tone come from the generated character. Face-focus renders go through `generateArtWithFace`, which regenerates up to `FACE_ATTEMPTS` (3) times until `/inspect` finds a face ≥ `MIN_FACE_HEIGHT` px and ≤ `MAX_FACE_YAW`, otherwise returns the best attempt. The response includes `faceCheck: { attempts, usable }`.
- `POST /api/face-identity` `{ photos: [dataUrl...] }` — returns `{ embedding (512 numbers), used, photos: [{ index, status: ok|no-face|unreadable|different-person }] }`.
- `POST /api/merge-face` `{ image, embedding | faceImage, region? }` — swaps the face onto the art. `region` is `{ cx, cy, rx, ry }` in art pixels and picks (or, failing detection, defines) the target face. Without it, the largest detected face is used. Returns 422 with `code: 'no-target-face'` when the art has no detectable face.

## Frontend behavior

- One input handles both modes: URLs containing `archidekt.com` load a deck; anything else is treated as a single Scryfall card and rendered immediately.
- Deck mode shows a grid with one tile per unique card (deduped by `scryfallId`, with a ×N badge for quantity). Duplicates are not generated multiple times.
- "Start processing" runs client-side: it loops over tiles sequentially, calling `/api/render` with `scryfallId` for each. Tiles move Queued → Generating → done (thumbnail) or error. Sequential on purpose, because the image model is slow and rate-limited.
- "Stop after current card" sets a flag checked between cards. The card in progress finishes. "Resume processing" re-runs queued and failed tiles and skips finished ones.
- "Your face" panel: the user adds several photos of one person and a short description of their look (pre-filled from the detector's age/sex estimate); they're downscaled client-side and sent to `/api/face-identity`, and the resulting embedding is kept in memory. With "Automatically swap" checked (and placeholder art off), every render — single card or deck tile — is followed by a swap onto the largest face. Art with no face keeps its original image and gets a note instead of an error. The pre-swap art is kept as `data.originalImage` for "Revert to original art".
- The detail panel (`showResult`) shows card text, the prompt, and the face-swap controls (manual swap into a dragged region uses the same identity). It auto-follows the newest finished card until the user clicks a tile (`followLatest`), and clicking a finished tile opens it.
- Every card (deck tile or single card) is an "item" `{ card, state, data, error, el }`; `generateItem` renders, auto-swaps, and persists one. Failed tiles are clickable and show the error with a Retry button; "Regenerate art" in the detail panel makes new art for any card. A failed regeneration keeps the previous art and shows the error.
- On reload the page reopens the last deck or single card (`cardmaker.lastOpened` in localStorage) from saved data only, without generating anything. Deck tiles with saved art start as done. The face embedding and description are also kept in localStorage; the photos are not.

## Saved data

- `data/` (gitignored; override with `CARDMAKER_DATA_DIR`): `data/cards/{scryfallId}/art.png` (current art, swap included), `original.png` (pre-swap art, if any) and `meta.json` (card, prompt, faceNote, faceCheck, savedAt), plus `data/decks/{id}.json`, the last fetched deck list. `/api/deck` falls back to the deck file when Archidekt is unreachable. Art is served statically at `/data/cards/...`.

## Notes

- Not a git repository.
- The UI has not been browser-tested end to end for the deck-processing and face-swap flows. The server endpoints were tested with curl against a real Archidekt deck and a real render. The card-frame overlay (`composeCard`) was verified with Playwright against several real cards (single/multicolor, legendary, land, artifact, planeswalker, double-digit P/T).

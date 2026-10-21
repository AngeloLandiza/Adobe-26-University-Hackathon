# Format spec cards (researched 2026-10-21)

Purpose: what official platform guidance says about each of our 13 sizes. Machine-readable twin: `training-data/references/format_specs.json`.
Rule: only claims I fetched and read. "Unverified" marks things I could not confirm. Primary sources = Meta, Google, YouTube, LinkedIn help pages. Honest caveat: the official pages say little about *layout* (where logo/CTA go); most are size, file and safe-area rules. Layout positions below are therefore mostly absent, not guessed.

Sources (short names used below)
- META-REELS: https://www.facebook.com/business/ads-guide/update/image/instagram-reels
- META-SAFE: https://www.facebook.com/business/help/103816146375741 (Meta page "About text overlays and the safe zone for ads in Stories and Reels", id 980593475366490)
- META-FEED: https://www.facebook.com/business/ads-guide/update/image/instagram-feed
- YT-SPEC: https://support.google.com/youtube/answer/72431
- YT-TIPS: https://support.google.com/youtube/answer/12340300
- LI-PAGES: https://www.linkedin.com/help/linkedin/answer/a563309/image-specifications-for-your-linkedin-pages-and-career-pages
- ADSENSE: https://support.google.com/adsense/answer/6002621 (supported sizes and placement)
- G-IMG: https://support.google.com/google-ads/answer/1722096 (image ad requirements)
- G-RDA: https://support.google.com/google-ads/answer/9823397 (responsive display ad best practices; applies to uploaded *assets*, not strictly to fixed-size banners, but is Google's only creative-content guidance found)

## Display ad rules shared by the 7 banner sizes (728x90, 300x250, 160x600, 970x250, 320x50, 300x600, 970x90)
- G-IMG: all seven are supported Google image-ad sizes; GIF/JPG/PNG, max 150KB; animation up to 30 s, under 5 fps.
- G-RDA (asset guidance): make the product/service the focus; blank space not over 80% of the image; single image not collage; avoid text on top of images (can be unreadable at small sizes) and avoid logo overlaid on image; use a clear call to action; headline clear and simple, sentence case not all caps; no fake buttons (play/download) in the image; keep edges square, no borders baked in.
- IAB: the IAB New Ad Portfolio page is paywalled/unreadable here; the IAB file-weight figures (150 kB initial) came only from a search summary and are **unverified**. LEAN whitepaper URL 404'd. No IAB claims are used.

## Instagram post 1080x1080
- Context: feed image. Meta recommends 4:5 (1440x1800); 1:1 falls inside the accepted range but is not the recommended ratio (META-FEED). Recommended primary text 125 chars, headline 40 chars; JPG/PNG; max 30MB; min width 500px (META-FEED).
- Safe zones: none stated for feed. Arrangement, text min size: unverified / not stated.
- Engine note: prior that text is brief; 1:1 is "acceptable", 4:5 preferred, so square adaptations are not Meta's first choice.

## Instagram portrait 1080x1350
- 4:5 is Meta's recommended feed ratio, 1440x1800 recommended (same ratio) (META-FEED). Same text limits as above.
- Engine note: the best-supported feed format; no safe zone to enforce.

## Instagram story 1080x1920
- Context: full-screen vertical; UI sits over top and bottom.
- Safe zones: keep text/logos/key elements out of roughly top and bottom 14% (about 250 px) (META-SAFE). For Reels: about 14% top, 35% bottom, 6% each side, so profile icon and CTA do not cover content (META-REELS); recommended 1440x2560, 9:16 (META-REELS).
- A search snippet said Meta's Stories ads-guide pages use 20% bottom; I did not open those pages, so **unverified**.
- Engine note: strongest rule candidate. Hard exclusion bands: top 14%, bottom 14% (Stories) up to 35% (if also served as Reels), sides 6%.

## YouTube thumbnail 1280x720
- 16:9, JPG/PNG; YouTube recommends the largest image possible (3840x2160), min 640 px wide; mobile limit 2MB (YT-SPEC).
- Design: thumbnails display at various sizes so keep it simple, avoid overly complex designs and hard-to-read fonts; if text, keep it short and use an easy font; bold colour helps but too much overwhelms (YT-TIPS).
- Not stated: face placement, contrast ratio, text size, the timestamp overlay corner (**unverified**; widely repeated in blogs but not checked).
- Engine note: prior of one dominant subject plus very short text.

## LinkedIn banner 1584x396
- Official LinkedIn Page cover is 1512x256 min/recommended; PNG/JPEG; max 3MB (LI-PAGES). 1584x396 is the personal-profile banner size per secondary blogs only, **unverified** against LinkedIn.
- Cover is cropped per screen: keep key details central, away from edges especially lower-right; use limited text (LI-PAGES). Logo shown separately (400x400 recommended).
- Engine note: this is a very wide strip; centered content, empty lower-right corner, little text.

## Facebook cover 820x312
- No primary source retrieved (Facebook help URLs returned 404). Secondary blogs say 820x312 desktop / 640x360 mobile with side crop on mobile; **unverified**. No conventions recorded.

## Leaderboard 728x90 / Large leaderboard 970x90 / Billboard 970x250
- 728x90: top performer; works above main content and on forums (ADSENSE). 970x90: expands to 970x415, suits HD photo/video content (ADSENSE). 970x250: premium position above main content, limited advertiser supply (ADSENSE).
- Shared rules above apply. Layout of logo/CTA within the strip: not stated by any primary source.
- Engine note: the only usable rules are content-level (one focus, short text, one clear CTA, text/logo kept as separate elements).

## Medium rectangle 300x250 / Half page 300x600 / Wide skyscraper 160x600
- 300x250: top performer, embedded in text or at article end (ADSENSE). 300x600: top performer, fast-growing (ADSENSE). 160x600: sidebar placement (ADSENSE).
- Shared rules above apply. Arrangement: not stated by primary sources.

## Mobile banner 320x50
- Mobile-only; works well at bottom of a phone page (ADSENSE). Shared rules apply; minimum text size not stated (**unverified**).

## Common mistakes (only the sourced ones)
Blurry/over-filtered or bordered images, collages, text and logos pasted over images, fake buttons, over 80% blank space, all caps (G-RDA); overly complex thumbnails, hard-to-read fonts (YT-TIPS); key content in Story top/bottom bands (META-SAFE).

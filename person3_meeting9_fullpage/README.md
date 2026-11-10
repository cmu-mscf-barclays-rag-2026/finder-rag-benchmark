# Secondary full-page OCR control

This local-only supplement measures the effect of page representation after the primary transfer run revealed extensive page truncation. It does not replace or modify either frozen primary stage and creates no new statistical test family.

All 2,942 supplied page OCR texts are covered by the existing offset-preserving 224-token/32-overlap helper. Empty pages retain one empty chunk. The same pinned MiniLM CPU float32 encoder embeds each chunk; normalized cosine similarities are maximized within each page. Page ties are resolved by original corpus index. All 309 question vectors and saved Focused/Short vectors are reused. Original full-OCR BM25 remains unchanged.

```bash
.venv/bin/python person3_meeting9_fullpage/run.py prepare
.venv/bin/python person3_meeting9_fullpage/run.py index
.venv/bin/python person3_meeting9_fullpage/run.py baseline
# After both transfer stages' saved generation vectors are available:
.venv/bin/python person3_meeting9_fullpage/run.py evaluate
.venv/bin/python person3_meeting9_fullpage/run.py validate
```

Chunk vectors checkpoint every 512 chunks and are reused on resume. Source text spans, exact token coverage, actual encoder token counts, vector normalization, hashes and full top100 page rankings are retained. Empty or missing generation vectors are never imputed. No command makes an API request.

The outcome is a descriptive coverage control, not a new confirmatory prompt experiment. Max-chunk scoring also gives longer pages more chances to match. OCR can still omit visual information; no image encoder or final-answer reader is evaluated.

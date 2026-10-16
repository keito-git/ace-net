#!/usr/bin/env python3
"""
P1-E: Automated validation of the under-determinacy proxy.

"Absence of an explicit emotion word" is a coarse proxy for "external context
is needed." This study does not use human annotation, so the proxy is validated
with an automated LLM judge and the agreement is reported as a bounded
substitute for human validation.

For a balanced sample of GoEmotions test texts, the judge answers two yes/no
questions per text:
  - context_required: is EXTERNAL context needed to determine the writer's
    emotion, or is the text self-contained?
  - has_explicit_emotion_word: does the text contain an explicit emotion word?

We then compare:
  (1) proxy validity: the lexicon flag no_explicit_emotion (proxy for
      "context needed") vs the judge's context_required;
  (2) lexicon coverage: the lexicon flag vs the judge's has_explicit_emotion_word.

Agreement is reported as percent agreement, Cohen's kappa, and the 2x2 table.

The judge uses OpenAI (gpt-4o) when OPENAI_API_KEY is set, else OpenRouter
(openai/gpt-4o) when OPENROUTER_API_KEY is set. API keys are read from the
environment and never printed.

Output: results/revision/p1e_underdeterminacy_validation.{json,md}
"""
import os, sys, json, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.registry import get_loader
from src.utils.stratify import compute_strata_flags

HERE    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(HERE, "results", "revision")
N_PER_GROUP = 150     # sampled per proxy group (no-emo-word vs has-emo-word)
SEED    = 42
MAX_WORKERS = 8
os.makedirs(OUT_DIR, exist_ok=True)

# Judge endpoint selection from environment variables.
if os.environ.get("OPENAI_API_KEY"):
    API_URL = "https://api.openai.com/v1/chat/completions"
    API_KEY = os.environ["OPENAI_API_KEY"]
    MODEL   = "gpt-4o"
    PROVIDER = "openai"
elif os.environ.get("OPENROUTER_API_KEY"):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY = os.environ["OPENROUTER_API_KEY"]
    MODEL   = "openai/gpt-4o"
    PROVIDER = "openrouter"
else:
    print("ERROR: set OPENAI_API_KEY or OPENROUTER_API_KEY in the environment.")
    sys.exit(1)

PROMPT = (
    "You are annotating short social-media texts for an emotion-recognition study.\n"
    "Text: \"\"\"{text}\"\"\"\n\n"
    "Answer two questions about THIS text:\n"
    "1. context_required: To correctly identify the WRITER'S emotion, is EXTERNAL "
    "context required (background about the situation, the conversation this replies "
    "to, or specific world knowledge not in the text)? If the emotion is determinable "
    "from the text alone, answer false.\n"
    "2. has_explicit_emotion_word: Does the text contain an explicit emotion word "
    "(e.g., happy, angry, sad, proud, afraid, grateful)?\n\n"
    "Respond ONLY with compact JSON: "
    '{{"context_required": true/false, "has_explicit_emotion_word": true/false}}'
)


def judge(text):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(text=text[:1500])}],
        "temperature": 0.0,
        "max_tokens": 40,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=40)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                d = json.loads(content)
                return bool(d.get("context_required")), bool(d.get("has_explicit_emotion_word")), None
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1)); continue
            return None, None, f"http{r.status_code}:{r.text[:120]}"
        except Exception as e:
            time.sleep(1.5 * (attempt + 1))
            last = str(e)[:120]
    return None, None, f"failed:{last}"


def kappa(a, b):
    """Cohen's kappa for two boolean lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else float("nan")


def confusion(proxy, judge_flag):
    """Map proxy/judge_flag boolean lists to 2x2 counts keyed by (proxy, judge)."""
    c = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
    for p, j in zip(proxy, judge_flag):
        c[("T" if p else "F") + ("T" if j else "F")] += 1
    return c


def main():
    loader = get_loader("goemotions")
    _, _, test_records = loader.load_splits()
    texts = [r["text"] for r in test_records]
    flags = compute_strata_flags(texts)
    no_emo = flags["no_explicit_emotion"]          # True = proxy says context-needed

    rng = random.Random(SEED)
    idx_no  = [i for i, v in enumerate(no_emo) if v]
    idx_yes = [i for i, v in enumerate(no_emo) if not v]
    rng.shuffle(idx_no); rng.shuffle(idx_yes)
    sample = idx_no[:N_PER_GROUP] + idx_yes[:N_PER_GROUP]
    print(f"[P1-E] provider={PROVIDER} model={MODEL} | sample={len(sample)} "
          f"(no-emo {min(N_PER_GROUP,len(idx_no))} + has-emo {min(N_PER_GROUP,len(idx_yes))})")

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(judge, texts[i]): i for i in sample}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            cr, hw, err = fut.result()
            results[i] = {"context_required": cr, "has_emo_word_judge": hw, "err": err}
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(sample)} judged")

    ok = [i for i in sample if results[i]["err"] is None and results[i]["context_required"] is not None]
    errs = [results[i]["err"] for i in sample if results[i]["err"]]
    proxy_ctx   = [no_emo[i] for i in ok]                                 # proxy: no emo word => context needed
    judge_ctx   = [results[i]["context_required"] for i in ok]
    judge_word  = [results[i]["has_emo_word_judge"] for i in ok]
    proxy_word  = [not no_emo[i] for i in ok]                             # proxy: has emo word

    n = len(ok)
    agree_ctx = sum(1 for a, b in zip(proxy_ctx, judge_ctx) if a == b) / n if n else float("nan")
    agree_word = sum(1 for a, b in zip(proxy_word, judge_word) if a == b) / n if n else float("nan")
    flagged = [j for p, j in zip(proxy_ctx, judge_ctx) if p]
    proxy_precision = sum(flagged) / len(flagged) if flagged else float("nan")

    report = {
        "provider": PROVIDER, "model": MODEL, "seed": SEED,
        "n_sampled": len(sample), "n_valid": n, "n_errors": len(errs),
        "error_examples": errs[:5],
        "proxy_validity_context_needed": {
            "percent_agreement": agree_ctx,
            "cohens_kappa": kappa(proxy_ctx, judge_ctx),
            "confusion_proxy_vs_judge": confusion(proxy_ctx, judge_ctx),
            "proxy_precision_for_context_need": proxy_precision,
        },
        "lexicon_coverage_emotion_word": {
            "percent_agreement": agree_word,
            "cohens_kappa": kappa(proxy_word, judge_word),
            "confusion_proxy_vs_judge": confusion(proxy_word, judge_word),
        },
    }
    with open(os.path.join(OUT_DIR, "p1e_underdeterminacy_validation.json"), "w") as f:
        json.dump(report, f, indent=2)

    pv = report["proxy_validity_context_needed"]
    lc = report["lexicon_coverage_emotion_word"]
    md = [
        "# P1-E Under-determinacy proxy validation (automated LLM judge)\n",
        f"Judge: {PROVIDER}/{MODEL} | valid responses: {n}/{len(sample)} (errors {len(errs)})\n",
        "## Proxy validity: no-emotion-word (proxy) vs judge 'context required'\n",
        f"- percent agreement = {pv['percent_agreement']*100:.1f}%",
        f"- Cohen's kappa = {pv['cohens_kappa']:.3f}",
        f"- proxy precision (of proxy-flagged, share judge also deems context-needed) = {pv['proxy_precision_for_context_need']*100:.1f}%",
        f"- confusion (proxy,judge) TT/TF/FT/FF = {pv['confusion_proxy_vs_judge']}",
        "\n## Lexicon coverage: lexicon emotion-word flag vs judge's emotion-word flag\n",
        f"- percent agreement = {lc['percent_agreement']*100:.1f}%",
        f"- Cohen's kappa = {lc['cohens_kappa']:.3f}",
        f"- confusion (proxy,judge) TT/TF/FT/FF = {lc['confusion_proxy_vs_judge']}",
        "\nNote: the LLM judge is itself imperfect; this is a bounded automated "
        "substitute for human validation, reported transparently.",
    ]
    with open(os.path.join(OUT_DIR, "p1e_underdeterminacy_validation.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSaved: {OUT_DIR}/p1e_underdeterminacy_validation.json and .md")


if __name__ == "__main__":
    main()

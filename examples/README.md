# Examples

One real, unedited run, so a reader can see the output without installing
anything.

[`metadata-filtering.md`](metadata-filtering.md) — *"How do vector databases
handle metadata filtering?"*, produced with:

```bash
python main.py "How do vector databases handle metadata filtering?" --max-steps 3
```

Three research rounds, 14 sources, one repair round. Worth looking at:

- the **frontmatter**, which records the model, the number of rounds and every
  source URL the tools returned — the report is validated against exactly that
  list
- the **`## Open questions and conflicts`** section, where the agent says which
  trade-off it could not resolve instead of smoothing it over
- the **reference list**, where every entry survived validation, and the
  inline `[n]` markers, which all have a matching entry

It needed one repair round: the first draft did not pass validation. That is
the normal case for anything beyond a textbook question, and the reason the
repair stage exists.

To swap in a run of your own, copy any file out of `reports/` and update this
page. An imperfect run is the more interesting exhibit — if the agent hedged or
left a question open, keep it.

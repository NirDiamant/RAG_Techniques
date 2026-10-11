# Review instructions

RAG_Techniques is a teaching repository of RAG tutorial notebooks. Most
contributors are opening their first open source pull request. Review for
teaching quality and for the contribution guidelines, not production hardening.

## Contribution guidelines check

Every pull request that adds or changes a notebook, a runnable script, an image,
or the technique list or table in README.md must be checked against
CONTRIBUTING.md, using the `rag-techniques-contribution-guidelines` skill.

These are must-fix findings (Medium):

- A new technique notebook outside `all_rag_techniques/`.
- A new technique without BOTH a README list entry and a README table row.
- New README entries whose GitHub or Colab links do not point at the new
  notebook.
- A new technique whose README list entry and table row carry different
  numbers.
- A new notebook that starts with code instead of a title, an overview and an
  explanation of the technique.
- A code cell the PR adds without a markdown cell before it describing what the
  cell does.

These are recommendations (Low) and must not block a merge: the layout of the
new README list entry, a missing diagram,
missing comparison or references sections, leftover outputs, where the new
technique sits in the simplest-to-most-complex order, and runnable script
placement.

## Scope

- Judge only what the pull request adds or changes. The README numbering and
  many older notebooks already break some of these rules; never ask a
  contributor to fix them.
- Read notebooks as cells, and quote cell content rather than raw `.ipynb` JSON.
- Do not raise missing tests, type hints, docstrings, logging or production
  error handling. Tutorial code is deliberately simple.

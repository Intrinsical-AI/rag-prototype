`repogpt_code_units_v5.json` is the canonical RepoGPT v5 producer output for the
versioned source fixture in `repogpt_eval_repo/`. It exercises RAG import,
metadata filtering, retrieval, idempotency, and scope replacement without
requiring another checkout. The default gate validates the recorded payload;
it does not execute the current producer.

To exercise the live producer, explicitly set `REPOGPT_ROOT` to its prepared
checkout. The helper runs that checkout's interpreter over the versioned source
fixture with `--emit code-units --replace-scope --include-tests --repo-key
repogpt_eval_repo`; configured paths or producer failures fail the test instead
of skipping it.

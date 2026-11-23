# Reproduction for getzep/graphiti#2165

`repro_2165.py` runs six scenarios through the full `add_episode` path, then calls `remove_episode`. It checks the `invalid_at` of the Ivanov edge after the removal.

- The LLM client is scripted. It applies one rule: a room has one guest at a time.
- The embedder is a word hash.
- The script makes no network calls and needs no API keys.

## Run

Start Neo4j or FalkorDB:

```
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/testpass neo4j:5.26.2
docker run -d -p 6379:6379 falkordb/falkordb:latest
```

Run the script with the `graphiti_core` that you want to test on the import path:

```
python repro_2165.py neo4j --mode none
python repro_2165.py falkordb --mode replay
python repro_2165.py kuzu --mode none --only S1,S3
```

## Modes

| Mode | What runs after `add_episode` |
| --- | --- |
| `none` | `remove_episode` of the installed `graphiti_core` |
| `naive` | `remove_episode`, then the `--patched` undo from the issue |
| `replay` | `remove_episode`, then an in-memory prototype that records the contradiction judgments and computes the minimum again |

On `main`, use `replay` to see the proposed behavior. On the branch `feat/2165-undo-invalidations`, `none` runs the implementation.

## Results by the author

Neo4j 5.26.2 and FalkorDB gave the same results.

| Graphiti | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| `main` at 9b8a463, mode `none` | fail | pass | fail | fail | fail | pass |
| `main` at 9b8a463, mode `naive` | pass | fail | fail | fail | fail | fail |
| `main` at 9b8a463, mode `replay` | pass | pass | pass | pass | pass | pass |
| `feat/2165-undo-invalidations`, mode `none` | pass | pass | pass | pass | pass | pass |

The author did not test the `kuzu` option. The Kuzu driver is not on `main`. Use a release that includes `graphiti_core.driver.kuzu_driver`.

Each line of output is one JSON object with the end dates of all edges before and after the removal. The last line lists the prompts that the scripted client answered.

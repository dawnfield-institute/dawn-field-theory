---
name: dft-lore-sync
description: Keep the Lore knowledge graph in sync after changing dawn-field-theory code, experiments, or structure. Use when moving or renaming experiments and documents, after a reorganization, or when updating an FDO or typed node — including the body-replacement and truncation traps.
---

# Vault sync — mandatory after structural change

After any change that moves, renames, adds or removes experiments or documents, update the
affected Lore nodes **in the same piece of work**. Vault drift is the top source of stale
documentation, and a stale graph is worse than no graph because Bert answers from it.

Search the graph *before* starting a topic, too: `lore_search(query, grade?)`. Trust grades:
`curated`/`source` are authoritative, `reference` is useful but flagged for rebuild (the
physics lives here), `legacy` is GRIM-era history.

**kronos is retired. Never write through `kronos_*` tools.**

## Two stores, two different fields

This trips people up — a path change must be applied to whichever store holds it.

| Store | Path field | Shape |
|---|---|---|
| **FDO** | `source_paths` | `[{repo, path, type}]`, `path` is repo-relative — no `dawn-field-theory/` prefix |
| **Typed** | `slots.sources` | strings **with** the `dawn-field-theory/` prefix |

Update FDO frontmatter with `lore_update(id, fields={...})`; update typed slots with
`lore_update_slots`. Both merge — they do not replace the node.

## The two traps that silently destroy content

**1. `lore_update(body=...)` REPLACES the entire body.** There is no append mode. To edit a
body you must have the whole current body first.

**2. `lore_get` truncates bodies at 8KB.** So the obvious sequence — `lore_get`, edit the
returned body, `lore_update(body=...)` — silently commits a truncated record for any node
over 8KB.

Fetch the full body over the HTTP API before writing one back:

```
GET  https://home.griminfra.com/bert/lore/api/fdo/{id}
PUT  same endpoint
```

`milestone6-planning-seed` has a body of 8191 characters — one byte under the boundary.
Do not assume a node is small enough.

**If you only need to change frontmatter, use `fields=` and never send a body at all.**
That is the safe path and avoids both traps entirely.

## MCP calls to lore are sequential, never parallel

Issue them one at a time. Parallel calls against the lore MCP face are unreliable here.

Sessions also go stale after a container restart — reinitialize rather than retrying a
failing call.

## Verify the write actually landed

Read the node back and check for a string you just wrote. A successful tool response is not
proof the content is what you intended, especially after a body replacement.

For a bulk migration, verify *every* path resolves on disk afterward rather than trusting
the count of nodes updated.

## Endpoints

Lore runs on mesh-host CT106, `http://192.168.8.230:8103` (MCP at `/mcp`, mesh-bearer
header required — the workspace `.mcp.json` carries it), and through the platform edge at
`https://home.griminfra.com/bert/lore/api/`.

The git mirrors `lore-vault/` (FDO) and `lore-vault-typed/` (typed) are **rendered
mirrors**, committed by a 15-minute cron on CT106. Postgres is truth. A local mirror clone
that looks stale is usually just unpulled — check `git fetch && git status -sb` before
concluding the sync is broken.

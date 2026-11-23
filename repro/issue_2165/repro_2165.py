"""Reproduce getzep/graphiti#2165 through the real add_episode path.

Graph: local Neo4j (bolt://localhost:7687, neo4j/testpass), FalkorDB (localhost:6379),
or embedded Kuzu (only on releases that still ship graphiti_core.driver.kuzu_driver).
LLM: a scripted client. It answers each prompt from a fixed world model and never calls a model.
Embedder: a deterministic bag-of-words hash. No network calls, no API keys.

Usage:
    python repro_2165.py [neo4j|falkordb|kuzu] [--mode MODE] [--only S1,S3]

Run it with the graphiti_core that you want to test on the import path.

MODE:
    none    remove_episode of the installed graphiti_core only (default)
    naive   upstream remove_episode, then the naive undo from the issue (--patched)
    replay  upstream remove_episode, then the replay prototype in this file

Scenarios (one group_id each):
    S1 basic        ep1 A (Ivanov, 11-01); ep2 B (Petrova, 11-03) ends A.        remove ep2 -> A current
    S2 restated     S1 + ep3 restates B.                                          remove ep2 -> A stays ended at 11-03
    S3 chain        S1 + ep3 C (Sidorov, 11-05) ends B.                          remove ep2 -> A ends at 11-05
    S4 backfill     ep1 B (Petrova, 11-03); ep2 A (Ivanov, 11-01) arrives later. remove ep1 -> A current
    S5 stated end   ep1 A (Ivanov, 11-01 until 11-10); ep2 B (Petrova, 11-03).    remove ep2 -> A ends at 11-10
    S6 two closers  ep1 A; ep2 B (11-03); ep3 D (Kim, 11-02) also ends A.        remove ep2 -> A ends at 11-02
"""

import asyncio
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'
import logging  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logging.getLogger('neo4j').setLevel(logging.ERROR)

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.client import CrossEncoderClient  # noqa: E402
from graphiti_core.edges import EntityEdge  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient  # noqa: E402
from graphiti_core.llm_client import LLMClient, LLMConfig  # noqa: E402
from graphiti_core.nodes import EpisodeType, EpisodicNode  # noqa: E402
from graphiti_core.utils.maintenance import edge_operations  # noqa: E402

ROOM = 'Room 402'
RUN_ID = datetime.now().strftime('%H%M%S')


def dt(day: str) -> datetime:
    return datetime.fromisoformat(f'2026-{day}T00:00:00').replace(tzinfo=timezone.utc)


def iso(value: datetime | None) -> str | None:
    return value.strftime('%m-%d') if value else None


# World model: episode text -> facts. Each fact: (guest, valid_at, invalid_at stated in the text).
WORLD: dict[str, list[tuple[str, str, str | None]]] = {}
FACT_DATES: dict[str, tuple[str, str | None]] = {}


def guest_fact(guest: str) -> str:
    return f'{guest} is the guest in {ROOM}'


def say(text: str, guest: str, valid: str, invalid: str | None = None) -> str:
    WORLD[text] = [(guest, valid, invalid)]
    FACT_DATES[guest_fact(guest)] = (valid, invalid)
    return text


class ScriptedLLM(LLMClient):
    """Answer each prompt from WORLD. Unknown prompts raise, so the log shows every call."""

    calls: list[str] = []

    def __init__(self):
        super().__init__(LLMConfig(api_key='none', model='scripted'), cache=False)

    async def _generate_response(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError('use generate_response')

    async def generate_response(self, messages, response_model=None, prompt_name=None, **kwargs):
        ScriptedLLM.calls.append(prompt_name or '?')
        text = '\n'.join(m.content for m in messages)
        if prompt_name == 'extract_nodes_and_edges.extract_message':
            return self._extract(self._current(text))
        if prompt_name and prompt_name.startswith('extract_nodes.extract_'):
            if prompt_name in ('extract_nodes.extract_text', 'extract_nodes.extract_message', 'extract_nodes.extract_json'):
                return {'extracted_entities': self._extract(self._current(text))['extracted_entities']}
        if prompt_name == 'extract_edges.edge':
            edges = self._extract(self._current(text))['edges']
            for e in edges:
                e.update(self._ts(e['fact']))
            return {'edges': edges}
        if prompt_name == 'dedupe_edges.resolve_edge':
            return self._resolve_edge(text)
        if prompt_name == 'extract_edges.extract_timestamps':
            fact = self._new_fact(text)
            valid, invalid = FACT_DATES[fact]
            return {'valid_at': f'2026-{valid}T00:00:00Z', 'invalid_at': f'2026-{invalid}T00:00:00Z' if invalid else None}
        if prompt_name == 'extract_edges.extract_timestamps_batch':
            facts = [f for f in FACT_DATES if f in text]
            return {'timestamps': [self._ts(f) for f in facts]}
        if prompt_name == 'dedupe_nodes.nodes':
            ids = [int(i) for i in re.findall(r"'id': (\d+)", text)]
            names = re.findall(r"'name': '([^']*)'", text)
            return {'entity_resolutions': [{'id': i, 'name': n, 'duplicate_candidate_id': -1} for i, n in zip(ids, names)]}
        if response_model is not None:
            # Summaries and attributes: return the empty shape of the model.
            fields = response_model.model_fields
            return {k: ([] if 'list' in str(v.annotation) else '') for k, v in fields.items() if v.is_required()}
        raise AssertionError(f'unexpected prompt {prompt_name}')

    @staticmethod
    def _ts(fact: str) -> dict[str, Any]:
        valid, invalid = FACT_DATES[fact]
        return {'valid_at': f'2026-{valid}T00:00:00Z', 'invalid_at': f'2026-{invalid}T00:00:00Z' if invalid else None}

    @staticmethod
    def _current(text: str) -> str:
        for open_tag, close_tag in (('<CURRENT MESSAGE>', '</CURRENT MESSAGE>'), ('<CURRENT_MESSAGE>', '</CURRENT_MESSAGE>'), ('<TEXT>', '</TEXT>')):
            if open_tag in text:
                return text.split(open_tag)[1].split(close_tag)[0]
        return text

    @staticmethod
    def _extract(text: str) -> dict[str, Any]:
        episode_text = next(t for t in WORLD if t in text)
        entities = [{'name': ROOM, 'entity_type_id': 0}]
        edges = []
        for guest, _valid, _invalid in WORLD[episode_text]:
            entities.append({'name': guest, 'entity_type_id': 0})
            edges.append({
                'source_entity_name': ROOM,
                'target_entity_name': guest,
                'relation_type': 'HAS_GUEST',
                'fact': guest_fact(guest),
                'episode_indices': [0],
            })
        return {'extracted_entities': entities, 'edges': edges}

    @staticmethod
    def _new_fact(text: str) -> str:
        block = text.split('<NEW FACT>')[1].split('</NEW FACT>')[0] if '<NEW FACT>' in text else text
        return next(f for f in FACT_DATES if f in block)

    def _resolve_edge(self, text: str) -> dict[str, Any]:
        new = self._new_fact(text)
        existing = text.split('<EXISTING FACTS>')[1].split('</EXISTING FACTS>')[0]
        candidates = text.split('<FACT INVALIDATION CANDIDATES>')[1].split('</FACT INVALIDATION CANDIDATES>')[0]
        dup: list[int] = []
        contra: list[int] = []
        for block, may_dup in ((existing, True), (candidates, False)):
            for idx, fact in re.findall(r"'idx': (\d+), 'fact': '([^']*)'", block):
                if fact == new:
                    if may_dup:
                        dup.append(int(idx))
                elif fact.endswith(f'is the guest in {ROOM}'):
                    # One guest per room: a different guest contradicts.
                    contra.append(int(idx))
        return {'duplicate_facts': dup, 'contradicted_facts': contra}


class HashEmbedder(EmbedderClient):
    async def create(self, input_data):
        text = input_data[0] if isinstance(input_data, list) else input_data
        return self._vec(str(text))

    async def create_batch(self, input_data_list):
        return [self._vec(t) for t in input_data_list]

    @staticmethod
    def _vec(text: str) -> list[float]:
        v = [0.0] * 64
        for tok in re.findall(r'\w+', text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


class NoRank(CrossEncoderClient):
    async def rank(self, query, passages):
        return [(p, 1.0) for p in passages]


# ---------------------------------------------------------------------------
# Prototype: record contradiction verdicts at ingest time, replay them on removal.
# A real change would store these on the edge. Here they live in process memory.
# ---------------------------------------------------------------------------
VERDICTS: dict[str, set[str]] = {}  # edge uuid -> uuids of edges the LLM judged contradictory (both ways)
STATED_END: dict[str, datetime | None] = {}  # edge uuid -> invalid_at that the source text stated

_orig_apply = edge_operations._apply_temporal_invalidation


def _recording_apply(resolved_edge, candidates):
    STATED_END.setdefault(resolved_edge.uuid, resolved_edge.invalid_at)
    for c in candidates:
        STATED_END.setdefault(c.uuid, c.invalid_at)
        VERDICTS.setdefault(resolved_edge.uuid, set()).add(c.uuid)
        VERDICTS.setdefault(c.uuid, set()).add(resolved_edge.uuid)
    return _orig_apply(resolved_edge, candidates)


edge_operations._apply_temporal_invalidation = _recording_apply


def derived_invalid_at(edge: EntityEdge, partners: list[EntityEdge]) -> datetime | None:
    """invalid_at = min(stated end, valid_at of each live contradicting edge that starts later)."""
    ends = [STATED_END.get(edge.uuid)] + [
        p.valid_at for p in partners if p.valid_at and edge.valid_at and p.valid_at > edge.valid_at
    ]
    ends = [e for e in ends if e is not None]
    return min(ends) if ends else None


async def replay_after_remove(driver, deleted_edge_uuids: set[str]):
    affected = {u for d in deleted_edge_uuids for u in VERDICTS.get(d, set())} - deleted_edge_uuids
    for uuid in affected:
        VERDICTS[uuid] -= deleted_edge_uuids
        edge = await EntityEdge.get_by_uuid(driver, uuid)
        partners = await EntityEdge.get_by_uuids(driver, sorted(VERDICTS[uuid])) if VERDICTS[uuid] else []
        new_end = derived_invalid_at(edge, partners)
        if new_end != edge.invalid_at:
            edge.invalid_at = new_end
            edge.expired_at = None if new_end is None else datetime.now(timezone.utc)
            await edge.load_fact_embedding(driver)
            await edge.save(driver)


async def naive_after_remove(driver, removed: EpisodicNode, touched: list[EntityEdge]):
    # The --patched control from the issue.
    for e in touched:
        if removed.uuid not in e.episodes and e.invalid_at is not None:
            e.invalid_at = None
            e.expired_at = None
            await e.load_fact_embedding(driver)
            await e.save(driver)


# ---------------------------------------------------------------------------


async def add(g: Graphiti, group: str, name: str, text: str, day: str) -> EpisodicNode:
    res = await g.add_episode(
        name=name,
        episode_body=text,
        source=EpisodeType.text,
        source_description='front desk log',
        reference_time=dt(day),
        group_id=group,
    )
    return res.episode


async def remove(g: Graphiti, ep: EpisodicNode, mode: str):
    driver = g.driver
    stored = await EpisodicNode.get_by_uuid(driver, ep.uuid)
    touched = await EntityEdge.get_by_uuids(driver, stored.entity_edges)
    before = {e.uuid for e in await EntityEdge.get_by_group_ids(driver, [stored.group_id])}
    await g.remove_episode(ep.uuid)
    after = {e.uuid for e in await EntityEdge.get_by_group_ids(driver, [stored.group_id])}
    if mode == 'naive':
        await naive_after_remove(driver, stored, touched)
    elif mode == 'replay':
        await replay_after_remove(driver, before - after)
    return stored


async def state(g: Graphiti, group: str) -> dict[str, Any]:
    edges = await EntityEdge.get_by_group_ids(g.driver, [group])
    out = {}
    for e in sorted(edges, key=lambda e: e.valid_at or dt('01-01')):
        guest = e.fact.split(' is the guest')[0]
        out[guest] = {'valid': iso(e.valid_at), 'invalid': iso(e.invalid_at), 'expired': e.expired_at is not None}
    return out


def guest_invalid(st: dict[str, Any], guest: str) -> str | None:
    return st[guest]['invalid'] if guest in st else 'MISSING'


async def scenario(base_driver, sid: str, mode: str) -> dict[str, Any]:
    group = f'g2165-{sid}-{mode}-{RUN_ID}'
    # FalkorDB keeps each group_id in its own graph, so bind the driver to the group.
    driver = base_driver.clone(database=group) if hasattr(base_driver, 'clone') and 'Falkor' in type(base_driver).__name__ else base_driver
    g = Graphiti(graph_driver=driver, llm_client=ScriptedLLM(), embedder=HashEmbedder(), cross_encoder=NoRank())
    await g.build_indices_and_constraints()
    WORLD.clear()
    FACT_DATES.clear()
    if sid == 'S5':
        # The fact text is the same, so the stated end date must be the only Ivanov entry.
        ep_a_end = say('Front desk: Ivanov is the guest in room 402 from Nov 1 until Nov 10.', 'Ivanov', '11-01', '11-10')
    else:
        ep_a = say('Front desk: Ivanov is the guest in room 402 from Nov 1.', 'Ivanov', '11-01')
    ep_b = say('Front desk: Petrova is the guest in room 402 from Nov 3.', 'Petrova', '11-03')
    ep_b2 = say('Night shift: confirmed, Petrova is the guest in room 402 from Nov 3.', 'Petrova', '11-03')
    ep_c = say('Front desk: Sidorov is the guest in room 402 from Nov 5.', 'Sidorov', '11-05')
    ep_d = say('Front desk: Kim is the guest in room 402 from Nov 2.', 'Kim', '11-02')

    if sid == 'S1':
        await add(g, group, 'ep1', ep_a, '11-01')
        e2 = await add(g, group, 'ep2', ep_b, '11-03')
        expect = None
    elif sid == 'S2':
        await add(g, group, 'ep1', ep_a, '11-01')
        e2 = await add(g, group, 'ep2', ep_b, '11-03')
        await add(g, group, 'ep3', ep_b2, '11-04')
        expect = '11-03'
    elif sid == 'S3':
        await add(g, group, 'ep1', ep_a, '11-01')
        e2 = await add(g, group, 'ep2', ep_b, '11-03')
        await add(g, group, 'ep3', ep_c, '11-05')
        expect = '11-05'
    elif sid == 'S4':
        e2 = await add(g, group, 'ep1', ep_b, '11-03')
        await add(g, group, 'ep2', ep_a, '11-01')
        expect = None
    elif sid == 'S5':
        await add(g, group, 'ep1', ep_a_end, '11-01')
        e2 = await add(g, group, 'ep2', ep_b, '11-03')
        expect = '11-10'
    elif sid == 'S6':
        await add(g, group, 'ep1', ep_a, '11-01')
        e2 = await add(g, group, 'ep2', ep_b, '11-03')
        await add(g, group, 'ep3', ep_d, '11-04')
        expect = '11-02'
    else:
        raise ValueError(sid)

    before = await state(g, group)
    removed = await remove(g, e2, mode)
    after = await state(g, group)
    got = guest_invalid(after, 'Ivanov')
    return {
        'scenario': sid,
        'removed': removed.name,
        'before': before,
        'after': after,
        'ivanov_invalid_expected': expect,
        'ivanov_invalid_got': got,
        'pass': got == expect,
    }


async def main():
    provider = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'neo4j'
    mode = sys.argv[sys.argv.index('--mode') + 1] if '--mode' in sys.argv else 'none'
    only = sys.argv[sys.argv.index('--only') + 1].split(',') if '--only' in sys.argv else None
    if provider == 'neo4j':
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        driver = Neo4jDriver('bolt://localhost:7687', 'neo4j', 'testpass')
    elif provider == 'kuzu':
        # Not tested by the author. The Kuzu driver is not on main any more.
        from graphiti_core.driver.kuzu_driver import KuzuDriver  # type: ignore[import-not-found]

        driver = KuzuDriver(':memory:')
    else:
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(host='localhost', port=6379)
    results = []
    for sid in only or ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']:
        VERDICTS.clear()
        STATED_END.clear()
        r = await scenario(driver, sid, mode)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))
    print('SUMMARY', provider, mode, ' '.join(f"{r['scenario']}={'PASS' if r['pass'] else 'FAIL'}" for r in results))
    print('LLM prompts used:', sorted(set(ScriptedLLM.calls)))
    await driver.close()


if __name__ == '__main__':
    asyncio.run(main())

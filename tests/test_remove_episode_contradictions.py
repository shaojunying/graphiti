"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from tests.helpers_test import group_id

pytest_plugins = ('pytest_asyncio',)


def day(value: int) -> datetime:
    return datetime(2026, 11, value, tzinfo=timezone.utc)


@pytest.fixture
def mock_embedder():
    # The names of this file are not in the embeddings of tests.helpers_test.
    embedder = Mock(spec=EmbedderClient)
    embedder.create = AsyncMock(return_value=[0.1] * 384)
    return embedder


@pytest.fixture
def mock_llm_client():
    mock_llm = Mock(spec=LLMClient)
    mock_llm.config = Mock()
    mock_llm.model = 'test-model'
    mock_llm.small_model = 'test-small-model'
    mock_llm.temperature = 0.0
    mock_llm.max_tokens = 1000
    mock_llm.cache_enabled = False
    mock_llm.cache_dir = None
    mock_llm.generate_response = AsyncMock(return_value={'summaries': []})
    return mock_llm


class Room:
    """A graph where each episode states the guest of one room.

    Each fact contradicts the facts of the other guests. The contradiction records
    are what ingestion stores for the order of the calls to `fact`.
    """

    def __init__(self):
        self.room = EntityNode(name='Room 402', group_id=group_id, labels=['Entity'])
        self.nodes: list[EntityNode] = [self.room]
        self.episodes: list[EpisodicNode] = []
        self.edges: dict[str, EntityEdge] = {}
        self.mentions: list[EpisodicEdge] = []

    def episode(self, name: str) -> EpisodicNode:
        episode = EpisodicNode(
            name=name,
            group_id=group_id,
            labels=[],
            created_at=day(20),
            source=EpisodeType.text,
            source_description='front desk',
            content=name,
            valid_at=day(1),
        )
        self.episodes.append(episode)
        return episode

    def fact(
        self,
        guest: str,
        episodes: list[EpisodicNode],
        valid_at: datetime,
        invalid_at: datetime | None,
        contradicted_by: list[str] | None,
        stated_invalid_at: datetime | None = None,
    ) -> EntityEdge:
        node = EntityNode(name=guest, group_id=group_id, labels=['Entity'])
        self.nodes.append(node)
        edge = EntityEdge(
            source_node_uuid=self.room.uuid,
            target_node_uuid=node.uuid,
            created_at=day(20),
            name='HAS_GUEST',
            fact=f'{guest} is the guest in room 402',
            episodes=[episode.uuid for episode in episodes],
            valid_at=valid_at,
            invalid_at=invalid_at,
            expired_at=day(20) if invalid_at is not None else None,
            contradicted_by=contradicted_by,
            stated_invalid_at=stated_invalid_at,
            group_id=group_id,
        )
        self.edges[guest] = edge
        for episode in episodes:
            episode.entity_edges.append(edge.uuid)
            for mentioned in (self.room, node):
                self.mentions.append(
                    EpisodicEdge(
                        source_node_uuid=episode.uuid,
                        target_node_uuid=mentioned.uuid,
                        created_at=day(20),
                        group_id=group_id,
                    )
                )
        return edge

    def ended_by(self, episode: EpisodicNode, *guests: str) -> None:
        # add_episode lists the edges that an episode ended in its entity_edges.
        for guest in guests:
            episode.entity_edges.append(self.edges[guest].uuid)

    async def save(self, driver, embedder) -> None:
        for node in self.nodes:
            await node.generate_name_embedding(embedder)
        for edge in self.edges.values():
            await edge.generate_embedding(embedder)
        await add_nodes_and_edges_bulk(
            driver, self.episodes, self.mentions, self.nodes, list(self.edges.values()), embedder
        )


async def remove_and_read(graph_driver, mock_llm_client, mock_embedder, room, episode):
    await room.save(graph_driver, mock_embedder)
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=Mock(spec=CrossEncoderClient),
    )
    await graphiti.build_indices_and_constraints()
    await graphiti.remove_episode(episode.uuid)
    remaining = [edge.uuid for edge in room.edges.values()]
    return {edge.uuid: edge for edge in await EntityEdge.get_by_uuids(graph_driver, remaining)}


@pytest.mark.asyncio
async def test_remove_episode_revives_edge_that_the_episode_ended(
    graph_driver, mock_llm_client, mock_embedder
):
    room = Room()
    ep1, ep2 = room.episode('ep1'), room.episode('ep2')
    ivanov = room.fact('Ivanov', [ep1], day(1), day(3), [])
    petrova = room.fact('Petrova', [ep2], day(3), None, [ivanov.uuid])
    room.ended_by(ep2, 'Ivanov')

    edges = await remove_and_read(graph_driver, mock_llm_client, mock_embedder, room, ep2)

    assert petrova.uuid not in edges
    assert edges[ivanov.uuid].invalid_at is None
    assert edges[ivanov.uuid].expired_at is None


@pytest.mark.asyncio
async def test_remove_episode_keeps_end_when_another_episode_supports_the_closer(
    graph_driver, mock_llm_client, mock_embedder
):
    room = Room()
    ep1, ep2, ep3 = room.episode('ep1'), room.episode('ep2'), room.episode('ep3')
    ivanov = room.fact('Ivanov', [ep1], day(1), day(3), [])
    petrova = room.fact('Petrova', [ep2, ep3], day(3), None, [ivanov.uuid])
    room.ended_by(ep2, 'Ivanov')

    edges = await remove_and_read(graph_driver, mock_llm_client, mock_embedder, room, ep2)

    assert edges[petrova.uuid].episodes == [ep3.uuid]
    assert edges[ivanov.uuid].invalid_at == day(3)


@pytest.mark.asyncio
async def test_remove_episode_ends_edge_at_a_later_contradiction(
    graph_driver, mock_llm_client, mock_embedder
):
    room = Room()
    ep1, ep2, ep3 = room.episode('ep1'), room.episode('ep2'), room.episode('ep3')
    ivanov = room.fact('Ivanov', [ep1], day(1), day(3), [])
    petrova = room.fact('Petrova', [ep2], day(3), day(5), [ivanov.uuid])
    # The date checks skipped Ivanov for Sidorov, because Ivanov had ended on day 3.
    sidorov = room.fact('Sidorov', [ep3], day(5), None, [petrova.uuid, ivanov.uuid])
    room.ended_by(ep2, 'Ivanov')
    room.ended_by(ep3, 'Petrova')

    edges = await remove_and_read(graph_driver, mock_llm_client, mock_embedder, room, ep2)

    assert edges[ivanov.uuid].invalid_at == day(5)
    assert edges[sidorov.uuid].invalid_at is None
    assert edges[sidorov.uuid].contradicted_by == [ivanov.uuid]


@pytest.mark.asyncio
async def test_remove_episode_revives_backfilled_edge(graph_driver, mock_llm_client, mock_embedder):
    room = Room()
    ep1, ep2 = room.episode('ep1'), room.episode('ep2')
    petrova = room.fact('Petrova', [ep1], day(3), None, [])
    # ep2 arrived later with an earlier fact. Petrova starts later, so it ended Ivanov.
    ivanov = room.fact('Ivanov', [ep2], day(1), day(3), [petrova.uuid])

    edges = await remove_and_read(graph_driver, mock_llm_client, mock_embedder, room, ep1)

    assert edges[ivanov.uuid].invalid_at is None
    assert edges[ivanov.uuid].contradicted_by == []


@pytest.mark.asyncio
async def test_remove_episode_restores_stated_end(graph_driver, mock_llm_client, mock_embedder):
    room = Room()
    ep1, ep2 = room.episode('ep1'), room.episode('ep2')
    ivanov = room.fact('Ivanov', [ep1], day(1), day(3), [], stated_invalid_at=day(10))
    room.fact('Petrova', [ep2], day(3), None, [ivanov.uuid])
    room.ended_by(ep2, 'Ivanov')

    edges = await remove_and_read(graph_driver, mock_llm_client, mock_embedder, room, ep2)

    assert edges[ivanov.uuid].invalid_at == day(10)
    assert edges[ivanov.uuid].expired_at is not None


@pytest.mark.asyncio
async def test_remove_episode_keeps_earliest_remaining_end(
    graph_driver, mock_llm_client, mock_embedder
):
    room = Room()
    ep1, ep2, ep3 = room.episode('ep1'), room.episode('ep2'), room.episode('ep3')
    ivanov = room.fact('Ivanov', [ep1], day(1), day(2), [])
    petrova = room.fact('Petrova', [ep2], day(3), None, [ivanov.uuid])
    kim = room.fact('Kim', [ep3], day(2), day(3), [ivanov.uuid, petrova.uuid])
    room.ended_by(ep2, 'Ivanov')
    room.ended_by(ep3, 'Ivanov')

    edges = await remove_and_read(graph_driver, mock_llm_client, mock_embedder, room, ep2)

    assert edges[ivanov.uuid].invalid_at == day(2)
    assert edges[kim.uuid].invalid_at is None


@pytest.mark.asyncio
async def test_remove_episode_keeps_edge_without_contradiction_record(
    graph_driver, mock_llm_client, mock_embedder
):
    room = Room()
    ep1, ep2 = room.episode('ep1'), room.episode('ep2')
    ivanov = room.fact('Ivanov', [ep1], day(1), day(3), None)
    room.fact('Petrova', [ep2], day(3), None, [ivanov.uuid])
    room.ended_by(ep2, 'Ivanov')

    edges = await remove_and_read(graph_driver, mock_llm_client, mock_embedder, room, ep2)

    assert edges[ivanov.uuid].invalid_at == day(3)
    assert edges[ivanov.uuid].contradicted_by is None

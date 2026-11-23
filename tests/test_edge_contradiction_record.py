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

from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from tests.helpers_test import group_id

pytest_plugins = ('pytest_asyncio',)


@pytest.fixture
def mock_embedder():
    # The names of this file are not in the embeddings of tests.helpers_test.
    embedder = Mock(spec=EmbedderClient)
    embedder.create = AsyncMock(return_value=[0.1] * 384)
    return embedder


@pytest.mark.asyncio
@pytest.mark.parametrize('bulk', [False, True])
async def test_contradiction_record_round_trip(graph_driver, mock_embedder, bulk):
    room = EntityNode(name='Room 402', group_id=group_id, labels=['Entity'])
    guest = EntityNode(name='Ivanov', group_id=group_id, labels=['Entity'])
    for node in (room, guest):
        await node.generate_name_embedding(mock_embedder)
    stated = datetime(2026, 11, 10, tzinfo=timezone.utc)
    recorded = EntityEdge(
        source_node_uuid=room.uuid,
        target_node_uuid=guest.uuid,
        name='HAS_GUEST',
        fact='Ivanov is the guest in room 402',
        group_id=group_id,
        created_at=datetime(2026, 11, 20, tzinfo=timezone.utc),
        valid_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        invalid_at=stated,
        contradicted_by=['edge-a', 'edge-b'],
        stated_invalid_at=stated,
    )
    legacy = EntityEdge(
        source_node_uuid=guest.uuid,
        target_node_uuid=room.uuid,
        name='STAYS_IN',
        fact='Ivanov stays in room 402',
        group_id=group_id,
        created_at=datetime(2026, 11, 20, tzinfo=timezone.utc),
    )
    # A new edge without contradictions has an empty record, which is not None.
    empty = EntityEdge(
        source_node_uuid=room.uuid,
        target_node_uuid=guest.uuid,
        name='CLEANED_BY',
        fact='Room 402 is cleaned by Ivanov',
        group_id=group_id,
        created_at=datetime(2026, 11, 20, tzinfo=timezone.utc),
        contradicted_by=[],
    )
    edges = [recorded, legacy, empty]
    for edge in edges:
        await edge.generate_embedding(mock_embedder)

    if bulk:
        await add_nodes_and_edges_bulk(graph_driver, [], [], [room, guest], edges, mock_embedder)
    else:
        for node in (room, guest):
            await node.save(graph_driver)
        for edge in edges:
            await edge.save(graph_driver)

    stored = {
        edge.uuid: edge
        for edge in await EntityEdge.get_by_uuids(graph_driver, [edge.uuid for edge in edges])
    }

    assert stored[recorded.uuid].contradicted_by == ['edge-a', 'edge-b']
    assert stored[recorded.uuid].stated_invalid_at == stated
    assert 'contradicted_by' not in stored[recorded.uuid].attributes
    assert stored[legacy.uuid].contradicted_by is None
    assert stored[legacy.uuid].stated_invalid_at is None
    assert stored[empty.uuid].contradicted_by == []

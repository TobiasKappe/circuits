import pytest

from circuits import Node
from circuits.exceptions import ContentionException


class TestNode:
    def test_copy(self):
        first = Node([True])
        second = Node([first])
        first.connections.append(second)

        changed = set(first.propagate(set()))

        assert changed == {second}
        assert second.value == [True]
        assert second.upstream is first

    def test_transitive(self):
        first = Node([True])
        second = Node(connections=[first])
        third = Node(connections=[second])
        first.connections.append(second)
        second.connections.append(third)

        changed = set(first.propagate(set()))

        assert changed == {second, third}
        assert second.value == [True]
        assert second.upstream is first
        assert third.value == [True]
        assert third.upstream is second

    def test_fork(self):
        first = Node([True])
        second = Node(connections=[first])
        third = Node(connections=[first])
        first.connections.append(second)
        first.connections.append(third)

        changed = set(first.propagate(set()))

        assert changed == {second, third}
        assert second.value == [True]
        assert second.upstream is first
        assert third.value == [True]
        assert third.upstream is first

    def test_contention(self):
        first = Node([True])
        second = Node()
        third = Node(connections=[first, second])
        first.connections.append(third)
        second.connections.append(third)

        first.value = [True]
        changed = set(first.propagate(set()))

        assert changed == {second, third}
        assert second.value == [True]
        assert second.upstream is third
        assert third.value == [True]
        assert third.upstream is first

        second.value = False
        with pytest.raises(ContentionException):
            changed = list(second.propagate(set()))

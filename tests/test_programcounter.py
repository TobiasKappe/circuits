import pytest

from circuits import Node
from circuits.pc import ProgramCounterElement


class TestProgramCounterElement:
    def test_is_resolvable(self):
        dist = Node(bitwidth=32)
        jump = Node()
        reset = Node()
        ctr = Node(bitwidth=32)
        clock = Node()

        counter = ProgramCounterElement(dist, jump, reset, ctr, clock)
        assert counter.is_resolvable()

    def test_reset(self):
        dist = Node(bitwidth=32)
        jump = Node()
        reset = Node()
        ctr = Node(bitwidth=32)
        clock = Node()

        counter = ProgramCounterElement(dist, jump, reset, ctr, clock)
        counter.value = 0x123

        # If reset is high, the value becomes zero.
        reset.value = [True]
        nodes = list(counter.resolve())
        assert counter.value == 0
        assert ctr in nodes
        assert ctr.value == [False]*32

        # A raised clock edge does not increase the counter
        clock.value = [False]
        nodes = list(counter.resolve())
        assert counter.value == 0
        assert ctr in nodes
        assert ctr.value == [False]*32

        clock.value = [True]
        nodes = list(counter.resolve())
        assert counter.value == 0
        assert ctr in nodes
        assert ctr.value == [False]*32

        # De-asserting the reset line keeps the value at zero.
        reset.value = [False]
        nodes = list(counter.resolve())
        assert counter.value == 0
        assert ctr in nodes
        assert ctr.value == [False]*32

    @pytest.mark.parametrize(
        'jump_value',
        [
            [[None]],
            [[False]],
        ]
    )
    def test_increment(self, jump_value):
        dist = Node(bitwidth=32)
        jump = Node(jump_value)
        reset = Node()
        ctr = Node(bitwidth=32)
        clock = Node()

        counter = ProgramCounterElement(dist, jump, reset, ctr, clock)
        assert counter.value == 0

        # Start with a low clock; no change to the value
        clock.value = [False]
        nodes = list(counter.resolve())
        assert counter.value == 0
        assert ctr in nodes
        assert ctr.value == [False]*32

        # Raise the clock edge; this increments the value by 4
        clock.value = [True]
        nodes = list(counter.resolve())
        assert counter.value == 4
        assert ctr in nodes
        assert ctr.value == [False, False, True] + [False]*29

        # Lower the clock edge again; this does not change the value
        clock.value = [False]
        nodes = list(counter.resolve())
        assert counter.value == 4
        assert ctr in nodes
        assert ctr.value == [False, False, True] + [False]*29

    def test_jump(self):
        dist = Node(bitwidth=32)
        jump = Node()
        reset = Node()
        ctr = Node(bitwidth=32)
        clock = Node()

        counter = ProgramCounterElement(dist, jump, reset, ctr, clock)
        counter.value = 0x10

        # A raised clock edge with an undefined value for dist does not
        # change the value of the counter
        jump.value = [True]

        clock.value = [False]
        nodes = list(counter.resolve())
        assert counter.value == 0x10
        assert ctr in nodes
        assert ctr.value == [False]*4 + [True] + [False]*27

        clock.value = [True]
        nodes = list(counter.resolve())
        assert counter.value == 0x10
        assert ctr in nodes
        assert ctr.value == [False]*4 + [True] + [False]*27

        # A raised clock edge with a value for dist increments the counter
        # by that much
        dist.value = [False]*5 + [True] + [False]*26

        clock.value = [False]
        nodes = list(counter.resolve())
        assert counter.value == 0x10
        assert ctr in nodes
        assert ctr.value == [False]*4 + [True] + [False]*27

        clock.value = [True]
        nodes = list(counter.resolve())
        assert counter.value == 0x30
        assert ctr in nodes
        assert ctr.value == [False]*4 + [True, True] + [False]*26

        # We can also decrement via two's complement.
        dist.value = [False]*4 + [True]*28

        clock.value = [False]
        nodes = list(counter.resolve())
        assert counter.value == 0x30
        assert ctr in nodes
        assert ctr.value == [False]*4 + [True, True] + [False]*26

        clock.value = [True]
        nodes = list(counter.resolve())
        assert counter.value == 0x20
        assert ctr in nodes
        assert ctr.value == [False]*5 + [True] + [False]*26

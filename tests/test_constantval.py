import pytest

from circuits import Node
from circuits.constant import \
    ConstantValElement, GroundElement, PowerElement


class TestConstantValElement:
    def test_resolve(self):
        output = Node()
        element = ConstantValElement(
            [True, False, True],
            output,
            bitwidth=3,
        )

        nodes = list(element.resolve())
        assert nodes == [output]
        assert output.value == [True, False, True]


class TestFixedValElement:
    @pytest.mark.parametrize(
        'bitwidth, cls, value',
        [
            [1, GroundElement, [False]],
            [2, GroundElement, [False, False]],
            [1, PowerElement, [True]],
            [2, PowerElement, [True, True]],
        ]
    )
    def test_resolve(self, bitwidth, cls, value):
        output = Node()
        element = cls(output, bitwidth=bitwidth)

        nodes = list(element.resolve())
        assert nodes == [output]
        assert output.value == value

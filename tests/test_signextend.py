import pytest

from ouca.circuits import Node
from ouca.circuits.sign import SignExtendElement


class TestSignExtendElement:
    @pytest.mark.parametrize(
        'input_val, resolvable',
        [
            [None, False],
            [[False], True],
            [[True], True],
        ]
    )
    def test_is_resolvable(self, input_val, resolvable):
        input = Node(input_val)
        output = Node()
        element = SignExtendElement(input, output)

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'input_val, output_val',
        [
            [[False], [False] * 32],
            [[True], [True] * 32],
            [[False, False], [False] * 32],
            [[True, False], [True] + [False] * 31],
            [[False, True], [False] + [True] * 31],
        ]
    )
    def test_resolve(self, input_val, output_val):
        bitwidth = len(input_val)
        input = Node(input_val, bitwidth=bitwidth)
        output = Node()
        element = SignExtendElement(input, output, bitwidth=bitwidth)

        nodes = list(element.resolve())

        assert nodes == [output]
        assert output.value == output_val

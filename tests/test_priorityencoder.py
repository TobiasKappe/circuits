import pytest

from circuits import Node
from circuits.coder import PriorityEncoderElement


class TestPriorityEncoderElement:
    @pytest.mark.parametrize(
        'input_vals, resolvable',
        [
            [[None, None], False],
            [[None, [True]], False],
            [[[False], [True]], True]
        ]
    )
    def test_is_resolvable(self, input_vals, resolvable):
        bits = len(input_vals).bit_length()-1

        inputs = [Node(input_val) for input_val in input_vals]
        outputs = [Node() for _ in range(bits)]
        enable = Node()
        element = PriorityEncoderElement(
            inputs,
            outputs,
            enable,
            bitwidth=bits
        )

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'input_vals, enable_val, output_vals',
        [
            [[False, False], False, None],
            [[False, True], True, [True]],
            [[True, False], True, [False]],
            [[True, True], True, [True]],
            [[False, False, True, False], True, [False, True]],
        ]
    )
    def test_resolve(self, input_vals, enable_val, output_vals):
        bits = len(input_vals).bit_length()-1

        inputs = [Node([input_val]) for input_val in input_vals]
        outputs = [Node() for _ in range(bits)]
        enable = Node()
        element = PriorityEncoderElement(
            inputs,
            outputs,
            enable,
            bitwidth=bits
        )

        nodes = list(element.resolve())
        assert enable in nodes
        assert enable.value == [enable_val]

        if not enable_val:
            assert len(nodes) == 1
        else:
            assert len(nodes) == 1 + len(output_vals)
            for i, output_val in enumerate(output_vals):
                assert outputs[i].value == [output_val]

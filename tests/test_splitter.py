import pytest

from circuits import Node
from circuits.splitter import SplitterElement


class TestSplitterElement:
    @pytest.mark.parametrize(
        'input_val, output_vals, resolvable',
        [
            [None, [None, None], False],
            [[True], [None, None], True],
            [None, [[True], None], False],
            [None, [None, [True]], False],
            [None, [[True], [False]], True],
        ]
    )
    def test_is_resolvable(self, input_val, output_vals, resolvable):
        input_node = Node(input_val)
        output_nodes = [Node(output_val) for output_val in output_vals]
        element = SplitterElement(
            input_node,
            output_nodes,
            2,
            [1, 1],
        )

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'bitwidth_in, bitwidths_out, input_val, output_vals',
        [
            [1, [1], [True], [[True]]],
            [
                4,
                [1, 2, 1],
                [True, True, True, True],
                [[True], [True, True], [True]]
            ],
            [
                5,
                [3, 1, 1],
                [False, False, False, False, False],
                [[False, False, False], [False], [False]]
            ],
            [
                5,
                [3, 1, 1],
                [True, False, True, False, True],
                [[True, False, True], [False], [True]]
            ],
        ]
    )
    def test_resolve_forward(
        self,
        bitwidth_in,
        bitwidths_out,
        input_val,
        output_vals
    ):
        assert len(bitwidths_out) == len(output_vals)

        input_node = Node(input_val)
        output_nodes = [Node() for _ in bitwidths_out]
        element = SplitterElement(
            input_node,
            output_nodes,
            bitwidth_in,
            bitwidths_out,
        )

        nodes = list(element.resolve())
        assert nodes == output_nodes
        for i, output_node in enumerate(output_nodes):
            assert output_node.value == output_vals[i]

    @pytest.mark.parametrize(
        'bitwidth_in, bitwidths_out, input_val, output_vals',
        [
            [1, [1], [True], [[True]]],
            [
                4,
                [1, 2, 1],
                [True, True, True, True],
                [[True], [True, True], [True]],
            ],
            [
                5,
                [3, 1, 1],
                [False, False, False, False, False],
                [[False, False, False], [False], [False]],
            ],
            [
                5,
                [3, 1, 1],
                [True, False, True, False, True],
                [[True, False, True], [False], [True]],
            ],
        ]
    )
    def test_resolve_backward(
        self,
        bitwidth_in,
        bitwidths_out,
        input_val,
        output_vals
    ):
        assert len(bitwidths_out) == len(output_vals)

        input_node = Node()
        output_nodes = [Node(val) for val in output_vals]
        element = SplitterElement(
            input_node,
            output_nodes,
            bitwidth_in,
            bitwidths_out,
        )

        nodes = list(element.resolve())
        assert nodes == [input_node]
        assert input_node.value == input_val

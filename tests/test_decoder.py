import pytest

from ouca.circuits import Node
from ouca.circuits.coder import DecoderElement


class TestDecoder:
    @pytest.mark.parametrize(
        'signal_size, input_vals, resolvable',
        [
            [1, None, False],
            [1, [None], False],
            [2, [False, False], True],
            [2, [True, False], True],
        ]
    )
    def test_is_resolvable(
        self,
        signal_size,
        input_vals,
        resolvable,
    ):
        input = Node()
        outputs = [Node() for _ in range(2**signal_size)]
        element = DecoderElement(signal_size, input, outputs)

        input.value = input_vals

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'signal_size, input_vals, output_vals',
        [
            [1, [False], [[True], [False]]],
            [1, [True], [[False], [True]]],
            [2, [False, False], [[True], [False], [False], [False]]],
            [2, [True, False], [[False], [True], [False], [False]]],
            [2, [False, True], [[False], [False], [True], [False]]],
            [2, [True, True], [[False], [False], [False], [True]]],
        ]
    )
    def test_resolve(
        self,
        signal_size,
        input_vals,
        output_vals,
    ):
        input = Node()
        outputs = [Node() for _ in range(2**signal_size)]
        element = DecoderElement(signal_size, input, outputs)

        input.value = input_vals

        nodes = list(element.resolve())

        for node in nodes:
            assert node in outputs

        for i, output in enumerate(outputs):
            assert output in nodes
            assert output.value == output_vals[i]

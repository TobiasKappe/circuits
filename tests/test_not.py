from ouca.circuits import sim

import pytest


class TestNotGateElement:
    @pytest.mark.parametrize(
        'input_val, resolvable',
        [
            [None, False],
            [False, True],
            [True, True],
        ]
    )
    def test_is_resolvable(self, input_val, resolvable):
        input_node = sim.Node(value=[input_val])
        output_node = sim.Node()
        element = sim.NotGateElement(input_node, output_node)

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'input_val, result',
        [
            [False, True],
            [True, False],
        ]
    )
    def test_resolve(self, input_val, result):
        input_node = sim.Node(value=[input_val])
        output_node = sim.Node()
        element = sim.NotGateElement(input_node, output_node)

        nodes = list(element.resolve())
        assert nodes == [output_node]
        assert output_node.value == [result]

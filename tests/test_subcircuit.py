from ouca.circuits import sim

import pytest


class TestSubCircuit:
    def test_copy_input(self):
        inner = [sim.InputElement(sim.Node(), None) for _ in range(2)]
        outer = [sim.Node() for _ in range(2)]
        sub = sim.Circuit(inner, sum((n.nodes for n in inner), start=[]))
        element = sim.SubCircuitElement(sub, outer, [])

        assert element.is_resolvable()

        outer[0].value = [False]
        outer[1].value = [True]
        nodes = list(element.resolve())
        assert nodes == []
        assert inner[0].state == [False]
        assert inner[1].state == [True]

    def test_copy_output(self):
        inner = [sim.OutputElement(sim.Node()) for _ in range(2)]
        outer = [sim.Node() for _ in range(2)]
        sub = sim.Circuit(inner, sum((n.nodes for n in inner), start=[]))
        element = sim.SubCircuitElement(sub, [], outer)

        assert element.is_resolvable()

        inner[0].inp1.value = [False]
        inner[1].inp1.value = [True]
        nodes = list(element.resolve())
        assert nodes == outer
        assert outer[0].value == [False]
        assert outer[1].value == [True]

    @pytest.mark.parametrize(
        'input_value, output_value',
        [
            [False, True],
            [True, False],
        ]
    )
    def test_compute(self, input_value, output_value):
        inner_input = sim.InputElement(sim.Node())
        inner_output = sim.OutputElement(sim.Node())
        inner_gate = sim.NotGateElement(inner_input.output1, inner_output.inp1)

        outer_input = sim.Node()
        outer_output = sim.Node()

        sub = sim.Circuit(
            [inner_input, inner_gate, inner_output],
            inner_input.nodes + inner_output.nodes + inner_gate.nodes,
        )
        element = sim.SubCircuitElement(sub, [outer_input], [outer_output])

        assert element.is_resolvable()

        outer_input.value = [input_value]
        nodes = list(element.resolve())
        assert nodes == [outer_output]
        assert outer_output.value == [output_value]

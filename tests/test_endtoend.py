import pytest

from pathlib import Path

from circuits import Circuit


class TestEndToEnd:
    CIRCUITS_PATH = Path(__file__).parent.parent.resolve() / 'exported'

    @pytest.mark.parametrize(
        'input_values, outcome, carry',
        [[[i == 1, j == 1, k == 1], (i+j+k) % 2 == 1, i+j+k > 1]
         for i in range(2)
         for j in range(2)
         for k in range(2)]
    )
    def test_adder(self, input_values, outcome, carry):
        circuit = Circuit.load_file(
            self.CIRCUITS_PATH / 'adder.cv',
            'Main'
        )

        for i, input_value in enumerate(input_values):
            circuit.inputs[i].state = [input_value]
        circuit.simulate()

        assert circuit.outputs[0].value == [outcome]
        assert circuit.outputs[1].value == [carry]

    @pytest.mark.parametrize(
        'input_values, output_index',
        [
            [[i == 1, j == 1], i*2+j]
            for i in range(2)
            for j in range(2)
        ]
    )
    def test_decoder(self, input_values, output_index):
        circuit = Circuit.load_file(
            self.CIRCUITS_PATH / 'decoder.cv',
            'Main'
        )

        for i, input_value in enumerate(input_values):
            circuit.inputs[i].state = [input_value]
        circuit.simulate()

        for i, output in enumerate(circuit.outputs):
            assert output.value == [3-i == output_index]

    def test_flipflop(self):
        circuit = Circuit.load_file(
            self.CIRCUITS_PATH / 'flipflop.cv',
            'Main'
        )

        circuit.inputs[0].state = [False]
        circuit.inputs[1].state = [True]
        circuit.simulate()

        assert circuit.outputs[0].value == [True]
        assert circuit.outputs[1].value == [False]

        circuit.inputs[0].state = [False]
        circuit.inputs[1].state = [False]
        circuit.simulate()

        assert circuit.outputs[0].value == [True]
        assert circuit.outputs[1].value == [False]

        circuit.inputs[0].state = [True]
        circuit.inputs[1].state = [False]
        circuit.simulate()

        assert circuit.outputs[0].value == [False]
        assert circuit.outputs[1].value == [True]

        circuit.inputs[0].state = [False]
        circuit.inputs[1].state = [False]
        circuit.simulate()

        assert circuit.outputs[0].value == [False]
        assert circuit.outputs[1].value == [True]

    @pytest.mark.parametrize(
        'input_values, outcome',
        [[[i == 1, j == 1, k == 1], (i <= j) <= (k > 0)]
         for i in range(2)
         for j in range(2)
         for k in range(2)]
    )
    def test_subcircuit(self, input_values, outcome):
        circuit = Circuit.load_file(
            self.CIRCUITS_PATH / 'subcircuit.cv',
            'Main'
        )

        for i, input_value in enumerate(input_values):
            circuit.inputs[i].state = [input_value]

        circuit.simulate()
        assert circuit.outputs[0].value == [outcome]

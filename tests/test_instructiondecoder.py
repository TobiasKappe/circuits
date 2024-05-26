from pathlib import Path

import pytest

from ouca.circuits import sim


class TestInstructionDecoder:
    CIRCUITS_PATH = Path(__file__).parent.parent.resolve() / 'circuits'

    @pytest.mark.parametrize(
        'opcode_val, add, addi, lw, sw, branch',
        [
            [[True, True, False, False, True, True, False],
             True, False, False, False, False],
            [[True, True, False, False, True, False, False],
             False, True, False, False, False],
            [[True, True, False, False, False, False, False],
             False, False, True, False, False],
            [[True, True, False, False, False, True, False],
             False, False, False, True, False],
            [[True, True, False, False, False, True, True],
             False, False, False, False, True],
            [[False, False, True, False, False, False, False],
             False, False, False, False, False]
        ]
    )
    def test_opcode(
        self,
        opcode_val,
        add,
        addi,
        lw,
        sw,
        branch
    ):
        circuit = sim.Circuit.load_file(
            self.CIRCUITS_PATH / 'instruction-decoder.cv',
            'opcode'
        )

        circuit.get_input('opcode').state = opcode_val
        circuit.simulate()

        assert circuit.get_output('add').value == [add]
        assert circuit.get_output('addi').value == [addi]
        assert circuit.get_output('lw').value == [lw]
        assert circuit.get_output('sw').value == [sw]
        assert circuit.get_output('branch').value == [branch]

    @pytest.mark.parametrize(
        'type, inp1, inp2, inp3, const',
        [
            [
                [False, False],
                [True, False, True, True, True, False, True],
                [True, False, True, False, True],
                [False, True, False, True, False],
                [False, False, True, False, True, True, False, True,
                 True, True, False, False, True, True, True, True,
                 True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True]
            ], [
                [False, False],
                [False, True, False, True, False, True, False],
                [True, True, False, True, True],
                [False, True, True, True, False],
                [False, False, True, True, True, False, True, False,
                 True, False, True, False, False, False, False, False,
                 False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False]
            ], [
                [True, False],
                [True, False, True, True, True, False, True],
                [True, False, True, False, True],
                [False, True, False, True, False],
                [True, False, True, False, True, True, False, True,
                 True, True, False, True, True, True, True, True,
                 True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True]
            ], [
                [False, True],
                [True, False, True, True, True, False, True],
                [True, False, True, False, True],
                [False, True, False, True, False],
                [True, False, True, False, True, True, False, True,
                 True, True, False, True, True, True, True, True,
                 True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True]
            ], [
                [True, False],
                [False, True, False, True, False, True, False],
                [True, True, False, True, True],
                [False, True, True, True, False],
                [True, True, False, True, True, False, True, False,
                 True, False, True, False, False, False, False, False,
                 False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False]
            ], [
                [False, True],
                [False, True, False, True, False, True, False],
                [True, True, False, True, True],
                [False, True, True, True, False],
                [True, True, False, True, True, False, True, False,
                 True, False, True, False, False, False, False, False,
                 False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False]
            ], [
                [True, True],
                [True, False, True, True, True, False, True],
                [True, False, True, False, True],
                [False, True, False, True, False],
                [False, True, False, True, False, True, False, True,
                 True, True, False, True, True, True, True, True,
                 True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True]
            ], [
                [True, True],
                [False, True, False, True, False, True, False],
                [True, True, False, True, True],
                [False, True, True, True, False],
                [False, True, True, True, False, False, True, False,
                 True, False, True, False, False, False, False, False,
                 False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False]
            ]
        ]
    )
    def test_swirl(self, type, inp1, inp2, inp3, const):
        circuit = sim.Circuit.load_file(
            self.CIRCUITS_PATH / 'instruction-decoder.cv',
            'swirl'
        )

        circuit.get_input('type').state = type
        circuit.get_input('inp1').state = inp1
        circuit.get_input('inp2').state = inp2
        circuit.get_input('inp3').state = inp3
        circuit.simulate()

        assert circuit.get_output('const').value == const

    @pytest.mark.parametrize(
        'instr, add, addi, lw, sw, blt, rd, rs1, rs2, const',
        [
            [
                [True, True, False, False, True, True, False, True,
                 True, True, False, False, False, False, False, False,
                 True, False, True, False, True, True, False, True,
                 True, False, False, False, False, False, False, False],
                True, False, False, False, False,
                [True, True, True, False, False],
                [False, True, False, True, False],
                [True, True, False, True, True],
                None,
            ], [
                [True, True, False, False, True, False, False, False,
                 False, False, True, False, False, False, False, True,
                 False, False, False, True, True, True, False, True,
                 True, False, True, True, False, True, True, False],
                False, True, False, False, False,
                [False, False, False, True, False],
                [True, False, False, False, True],
                None,
                [True, True, False, True, True, False, True, True,
                 False, True, True, False, False, False, False, False,
                 False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False],
            ], [
                [True, True, False, False, False, False, False, False,
                 True, True, True, False, False, True, False, False,
                 True, True, False, True, False, True, True, False,
                 True, True, False, True, True, False, True, True],
                False, False, True, False, False,
                [False, True, True, True, False],
                [False, True, True, False, True],
                None,
                [False, True, True, False, True, True, False, True,
                 True, False, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True],
            ], [
                [True, True, False, False, False, True, False, True,
                 False, True, False, True, False, True, False, True,
                 True, False, False, True, True, False, True, True,
                 True, False, True, False, True, False, True, False],
                False, False, False, True, False,
                None,
                [True, True, False, False, True],
                [True, False, True, True, True],
                [True, False, True, False, True, False, True, False,
                 True, False, True, False, False, False, False, False,
                 False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False],
            ], [
                [True, True, False, False, False, True, True, False,
                 False, False, True, False, False, False, True, True,
                 True, False, True, False, True, True, True, False,
                 True, False, False, True, False, False, True, True],
                False, False, False, False, True,
                None,
                [True, True, False, True, False],
                [True, True, True, False, True],
                [False, False, False, False, True, False, False, True,
                 False, False, True, False, True, True, True, True,
                 True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True],
            ]
        ]
    )
    def test_decode(
        self,
        instr,
        add,
        addi,
        lw,
        sw,
        blt,
        rd,
        rs1,
        rs2,
        const
    ):
        circuit = sim.Circuit.load_file(
            self.CIRCUITS_PATH / 'instruction-decoder.cv',
            'decode'
        )

        circuit.get_input('instr').state = instr
        circuit.simulate()

        assert circuit.get_output('add').value == [add]
        assert circuit.get_output('addi').value == [addi]
        assert circuit.get_output('lw').value == [lw]
        assert circuit.get_output('sw').value == [sw]
        assert circuit.get_output('blt').value == [blt]

        if rd is not None:
            assert circuit.get_output('rd').value == rd
        if rs1 is not None:
            assert circuit.get_output('rs1').value == rs1
        if rs2 is not None:
            assert circuit.get_output('rs2').value == rs2

        if const is not None:
            assert circuit.get_output('const').value == const

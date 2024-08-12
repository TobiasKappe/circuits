import pytest

from circuits import Node
from circuits.idec import InstructionDecoderElement


class TestInstructionDecoder:
    @pytest.mark.parametrize(
        'instr, resolvable', [
            [None, False],
            [[False]*32, True],
            [[False, True]*16, True],
            [[True]*31 + [None], False],
        ]
    )
    def test_is_resolvable(self, instr, resolvable):
        idec = InstructionDecoderElement(
            Node(bitwidth=32),
            Node(),
            Node(),
            Node(),
            Node(),
            Node(),
            Node(bitwidth=5),
            Node(bitwidth=5),
            Node(bitwidth=5),
            Node(bitwidth=32),
        )

        idec.instr.value = instr
        assert idec.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'instr, add, addi, lw, sw, blt, rd, rs1, rs2, constant',
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
                [False, False, False, True, False, False, False, True,
                 False, False, True, False, True, True, True, True,
                 True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True],
            ]
        ]
    )
    def test_resolve(
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
        constant,
    ):
        idec = InstructionDecoderElement(
            Node(bitwidth=32),
            Node(),
            Node(),
            Node(),
            Node(),
            Node(),
            Node(bitwidth=5),
            Node(bitwidth=5),
            Node(bitwidth=5),
            Node(bitwidth=32),
        )

        idec.instr.value = instr
        nodes = list(idec.resolve())

        assert idec.addi in nodes
        assert idec.add in nodes
        assert idec.lw in nodes
        assert idec.sw in nodes
        assert idec.blt in nodes

        assert idec.rd in nodes
        assert idec.rs1 in nodes
        assert idec.rs2 in nodes

        if constant is not None:
            assert idec.constant in nodes

        assert idec.addi.value == [addi]
        assert idec.add.value == [add]
        assert idec.lw.value == [lw]
        assert idec.sw.value == [sw]
        assert idec.blt.value == [blt]

        if rd is not None:
            assert idec.rd.value == rd
        if rs1 is not None:
            assert idec.rs1.value == rs1
        if rs2 is not None:
            assert idec.rs2.value == rs2

        if constant is not None:
            assert idec.constant.value == constant

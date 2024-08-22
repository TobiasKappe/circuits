from functools import singledispatchmethod
from enum import Enum

from circuits.element import Element
from circuits.mock import MockElement
from circuits.node import Node
from circuits.registry import ElementRegistry

InstructionType = Enum(
    'InstructionType',
    ['R_TYPE', 'I_TYPE', 'S_TYPE', 'B_TYPE']
)


class MockInstructionDecoderElement(MockElement):
    inputs = {
        'instr': (False, 32),
    }

    outputs = {
        'addi': 1,
        'add': 1,
        'lw': 1,
        'sw': 1,
        'blt': 1,
        'rd': 5,
        'rs1': 5,
        'rs2': 5,
        'constant': 32,
    }


@ElementRegistry.add_impl(
    'InstructionDecoder',
    MockInstructionDecoderElement
)
class InstructionDecoderElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        instr: Node,
        addi: Node,
        add: Node,
        lw: Node,
        sw: Node,
        blt: Node,
        rd: Node,
        rs1: Node,
        rs2: Node,
        constant: Node,
        **kwargs,
    ):
        self.instr = instr
        self.addi = addi
        self.add = add
        self.lw = lw
        self.sw = sw
        self.blt = blt
        self.rd = rd
        self.rs1 = rs1
        self.rs2 = rs2
        self.constant = constant

        super().__init__(
            [
                self.instr,
                self.addi,
                self.add,
                self.lw,
                self.sw,
                self.blt,
                self.rd,
                self.rs1,
                self.rs2,
                self.constant,
            ],
            **kwargs
        )

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)

    def is_resolvable(self):
        if self.instr.value is None:
            return False
        if None in self.instr.value:
            return False

        return True

    def resolve(self):
        opcode = self.instr.value[:7]

        instr_type = InstructionType.R_TYPE

        self.addi.value = [False]
        self.add.value = [False]
        self.lw.value = [False]
        self.sw.value = [False]
        self.blt.value = [False]

        if opcode == [True, True, False, False, True, False, False]:
            self.addi.value = [True]
            instr_type = InstructionType.I_TYPE

        if opcode == [True, True, False, False, True, True, False]:
            self.add.value = [True]
            instr_type = InstructionType.R_TYPE

        if opcode == [True, True, False, False, False, False, False]:
            self.lw.value = [True]
            instr_type = InstructionType.I_TYPE

        if opcode == [True, True, False, False, False, True, False]:
            self.sw.value = [True]
            instr_type = InstructionType.S_TYPE

        if opcode == [True, True, False, False, False, True, True]:
            self.blt.value = [True]
            instr_type = InstructionType.B_TYPE

        yield self.addi
        yield self.add
        yield self.lw
        yield self.sw
        yield self.blt

        self.rd.value = self.instr.value[7:12]
        self.rs1.value = self.instr.value[15:20]
        self.rs2.value = self.instr.value[20:25]

        yield self.rd
        yield self.rs1
        yield self.rs2

        if instr_type == InstructionType.I_TYPE:
            self.constant.value = (
                self.instr.value[20:32] +
                [self.instr.value[31]] * 20
            )
            yield self.constant
        elif instr_type == InstructionType.S_TYPE:
            self.constant.value = (
                self.instr.value[7:12] +
                self.instr.value[25:32] +
                [self.instr.value[31]] * 20
            )
            yield self.constant
        elif instr_type == InstructionType.B_TYPE:
            self.constant.value = (
                [False] +
                self.instr.value[8:12] +
                self.instr.value[25:31] +
                self.instr.value[7:8] +
                [self.instr.value[31]] * 20
            )
            yield self.constant

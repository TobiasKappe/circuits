from functools import singledispatchmethod
from typing import List, Union

from riscv.data import RiscInteger

from circuits.element import Element
from circuits.node import Node
from circuits.registry import ElementRegistry


@ElementRegistry.add_impl('RegisterFile')
class RegisterFileElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        R1: Node,
        R2: Node,
        R3: Node,
        clock: Node,
        dataIn: Node,
        dataOut1: Node,
        dataOut2: Node,
        en: Node,
        values: Union[List[RiscInteger], None] = None,
        **kwargs
    ):
        super().__init__(
            [R1, R2, R3, clock, dataIn, dataOut1, dataOut2, en],
            **kwargs
        )
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.clock = clock
        self.dataIn = dataIn
        self.dataOut1 = dataOut1
        self.dataOut2 = dataOut2
        self.en = en
        self.values = values or [RiscInteger(0) for _ in range(32)]
        self.prev_clock = None
        self.validate()

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.values = [
            RiscInteger(v, signed=False) if v >= 0 else RiscInteger(v)
            for v in raw_element['customData']['constructorParamaters'][0]
        ]
        self.prev_clock = None
        self.validate()

    def validate(self):
        assert len(self.values) == 32
        assert all(isinstance(v, RiscInteger) for v in self.values)

        assert self.R1.bitwidth == 5
        assert self.R2.bitwidth == 5
        assert self.R3.bitwidth == 5
        assert self.dataIn.bitwidth == 32
        assert self.en.bitwidth == 1
        assert self.clock.bitwidth == 1
        assert self.dataOut1.bitwidth == 32
        assert self.dataOut2.bitwidth == 32

    def is_resolvable(self):
        return True

    def resolve(self):
        if self.R3.value is not None and \
           self.dataIn.value is not None and \
           self.prev_clock is False and \
           self.clock.value is not None and \
           self.clock.value[0] is True and \
           self.en.value is not None and \
           self.en.value[0] is True and \
           any(self.R3.value):
            addr = RiscInteger(self.R3.value + [False] * 27)
            self.values[addr.to_int()] = RiscInteger(self.dataIn.value)

        if self.R1.value is not None:
            addr = RiscInteger(self.R1.value + [False] * 27)
            self.dataOut1.value = self.values[addr.to_int()].bits
            yield self.dataOut1

        if self.R2.value is not None:
            addr = RiscInteger(self.R2.value + [False] * 27)
            self.dataOut2.value = self.values[addr.to_int()].bits
            yield self.dataOut2

        if self.clock.value is not None:
            self.prev_clock = self.clock.value[0]

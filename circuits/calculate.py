from functools import singledispatchmethod

from circuits.element import Element
from circuits.node import Node
from circuits.registry import ElementRegistry


class BitArithmeticElement(Element):
    def half_add(self, x, y):
        return x != y, x and y

    def full_add(self, left, right, carry=False):
        carry = False
        outcomes = []

        for i in range(self.bitwidth):
            tmp_sum, carry_left = self.half_add(left[i], carry)
            outcome, carry_right = self.half_add(right[i], tmp_sum)
            carry = carry_left or carry_right
            outcomes.append(outcome)

        return outcomes, carry

    def twos_complement(self, bits):
        complemented, _ = self.full_add(
            [not b for b in bits],
            [True] + [False] * (self.bitwidth - 1)
        )
        return complemented

    def full_sub(self, left, right):
        return self.full_add(left, self.twos_complement(right))


@ElementRegistry.add_impl('ALU')
class ALUElement(BitArithmeticElement):
    @singledispatchmethod
    def __init__(
        self,
        inp1: Node,
        inp2: Node,
        controlSignalInput: Node,
        carryOut: Node,
        output: Node,
        **kwargs,
    ):
        self.inp1 = inp1
        self.inp2 = inp2
        self.controlSignalInput = controlSignalInput
        self.carryOut = carryOut
        self.output = output

        super().__init__(
            [
                self.inp1,
                self.inp2,
                self.controlSignalInput,
                self.carryOut,
                self.output,
            ],
            **kwargs
        )

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]

    def is_resolvable(self):
        if self.inp1.value is None or None in self.inp1.value:
            return False
        if self.inp2.value is None or None in self.inp2.value:
            return False
        if self.controlSignalInput.value is None or \
           None in self.controlSignalInput.value:
            return False

        # Undefined control value does not seem to change any output
        if self.controlSignalInput.value == [True, True, False]:
            return False

        return True

    def resolve(self):
        if self.controlSignalInput.value == [False, False, False]:
            self.carryOut.value = [False]
            self.output.value = [
                self.inp1.value[i] and self.inp2.value[i]
                for i in range(self.bitwidth)
            ]
        elif self.controlSignalInput.value == [True, False, False]:
            self.carryOut.value = [False]
            self.output.value = [
                self.inp1.value[i] or self.inp2.value[i]
                for i in range(self.bitwidth)
            ]
        elif self.controlSignalInput.value == [False, True, False]:
            self.output.value, carry = \
                self.full_add(self.inp1.value, self.inp2.value)
            self.carryOut.value = [carry]
        elif self.controlSignalInput.value == [False, False, True]:
            self.carryOut.value = [False]
            self.output.value = [
                self.inp1.value[i] and not self.inp2.value[i]
                for i in range(self.bitwidth)
            ]
        elif self.controlSignalInput.value == [True, False, True]:
            self.carryOut.value = [False]
            self.output.value = [
                self.inp1.value[i] or not self.inp2.value[i]
                for i in range(self.bitwidth)
            ]
        elif self.controlSignalInput.value == [False, True, True]:
            self.carryOut.value = [False]
            self.output.value, _ = \
                self.full_sub(self.inp1.value, self.inp2.value)
        elif self.controlSignalInput.value == [True, True, True]:
            self.carryOut.value = [False]
            sign1 = self.inp1.value[-1]
            sign2 = self.inp2.value[-1]
            if sign1 < sign2:
                outcome = False
            elif sign2 < sign1:
                outcome = True
            else:
                sub, _ = self.full_sub(self.inp1.value, self.inp2.value)
                outcome = sub[-1]
            self.output.value = [outcome] + [False] * (self.bitwidth - 1)

        yield self.carryOut
        yield self.output


@ElementRegistry.add_impl('Adder')
class AdderElement(BitArithmeticElement):
    @singledispatchmethod
    def __init__(
        self,
        inpA: Node,
        inpB: Node,
        carryIn: Node,
        carryOut: Node,
        sum: Node,
        **kwargs,
    ):
        self.inpA = inpA
        self.inpB = inpB
        self.carryIn = carryIn
        self.carryOut = carryOut
        self.sum = sum

        super().__init__(
            [
                self.inpA,
                self.inpB,
                self.carryIn,
                self.carryOut,
                self.sum,
            ],
            **kwargs
        )

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]

    def is_resolvable(self):
        if self.inpA.value is None or None in self.inpA.value:
            return False
        if self.inpB.value is None or None in self.inpB.value:
            return False

        return True

    def resolve(self):
        if self.carryIn.value is None or self.carryIn.value[0] is None:
            carry = False
        else:
            carry = self.carryIn.value[0]

        self.sum.value, carry = self.full_add(
            self.inpA.value,
            self.inpB.value,
            carry
        )
        self.carryOut.value = [carry]

        yield self.carryOut
        yield self.sum


@ElementRegistry.add_impl('TwoComplement')
class TwoComplementElement(BitArithmeticElement):
    @singledispatchmethod
    def __init__(
        self,
        inp1: Node,
        output1: Node,
        **kwargs
    ):
        self.inp1 = inp1
        self.output1 = output1

        super().__init__([self.inp1, self.output1], **kwargs)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]

    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        if None in self.inp1.value:
            return False
        return True

    def resolve(self):
        self.output1.value = self.twos_complement(self.inp1.value)
        yield self.output1

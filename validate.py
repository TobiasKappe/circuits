from ouca.circuits import sim

from argparse import ArgumentParser

import json


def bitstring_to_list(bitstring):
    return [v == "1" for v in bitstring][::-1]


def list_to_bitstring(list_):
    return ''.join(map(lambda x: '1' if x else '0', list_[::-1]))


def main():
    parser = ArgumentParser(
        prog='validate',
        description='Tests a circuit for given inputs',
    )
    parser.add_argument('circuit')
    parser.add_argument('inputs')

    args = parser.parse_args()

    with open(args.inputs, 'r') as handle:
        tests_per_subcircuit = json.load(handle)

    for subcircuit, subcircuit_tests in tests_per_subcircuit.items():
        for i, subcircuit_test in enumerate(subcircuit_tests):
            circuit = sim.Circuit.load_file(args.circuit, subcircuit)

            # Flush the values that were left in the circuit
            circuit.simulate()

            inputs = subcircuit_test['inputs'].items()
            for input_name, input_value in inputs:
                input_value = bitstring_to_list(input_value)
                circuit.get_input(input_name).state = input_value

            circuit.simulate()

            outputs = subcircuit_test['outputs'].items()
            for output_name, expected_value in outputs:
                if expected_value is None:
                    continue

                if circuit.get_output(output_name).value:
                    output_value = list_to_bitstring(
                        circuit.get_output(output_name).value
                    )
                else:
                    output_value = None

                if output_value != expected_value:
                    print(
                        f'[{subcircuit}/{i}] '
                        f'Output "{output_name}": '
                        f'expected "{expected_value}" '
                        f'but got "{output_value}"'
                    )


if __name__ == '__main__':
    main()

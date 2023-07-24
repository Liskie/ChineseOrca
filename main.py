from translator import OrcaTranslator

if __name__ == '__main__':

    translator = OrcaTranslator()

    translator.translate_instructions(num_lines=10)

    print(translator.instruction2answer)


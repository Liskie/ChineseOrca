from translator import OrcaTranslator, SupportedDatasetTypes

if __name__ == '__main__':

    translator = OrcaTranslator(SupportedDatasetTypes.Local)

    translator.translate_instructions(num_lines=10)


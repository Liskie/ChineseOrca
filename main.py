from translator import OrcaTranslator, SupportedDatasetType

if __name__ == '__main__':

    translator = OrcaTranslator(dataset_type=SupportedDatasetType.LismbpLocal,
                                num_datapoints_to_process=100,
                                num_workers=20)

    translator.translate_system_prompts()

    translator.translate_questions()

    translator.generate_responses()



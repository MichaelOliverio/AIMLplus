import os
from aiml_parser.parser import AIMLParser
from common.frame_extraction.frame_extractor import FrameExtractor
from common.frame_extraction.t5_frame_extractor_handler import T5FrameExtractorHandler
from nlu.nlu import NLU
from dm.dm import DM

if __name__ == "__main__":
    model_path = f'models\\Flan-T5-Base-Frame-Extractor'
    if not os.path.exists(os.path.abspath(model_path)):
        print(f"File non trovati/o")
        exit()

    test_folder = "aiml_data"
    if not os.path.exists(os.path.abspath(test_folder)):
        print(f"File non trovati/o")
        exit()

    frame_extractor = FrameExtractor(T5FrameExtractorHandler(model_path, 'google/flan-t5-base'))

    parser = AIMLParser()
    parser.load_from_folder(test_folder)

    nlu = NLU(frame_extractor)
    dm = DM(parser.categories, method="frame", uncertainty_threshold=0.3) # ricerca per frame 
    
    #nlu_output = {
    #    'user_input': 'tell me about automaton',
    #    'intent': 'fsa-practical',
    #    'argument': 'automata',
    #    'dialogue_act': 'Ta:request',
    #    'frame': {'transitions': ['q1','q2','0']}
    #}
    #action = dm.process_input(nlu_output)
    #print(f"response: {action.template}")

    while True:
        input_text = input("input: ")
        if input_text == "exit":
            break

        nlu_output = nlu.extraction(input_text)
        print(f"nlu_output: {nlu_output}")

        action = dm.process_input(nlu_output)
        print(f"response: {action.template}")
        print()
import sys

src_paths = [
    'c:\\...\\AIMLplus\\',
]

for src_path in src_paths:
    if src_path not in sys.path:
        sys.path.append(src_path)

from common.frame_extraction.base_frame_extractor_handler import BaseFrameExtractorHandler
from common.frame_extraction.frame_parser import FrameParser

class FrameExtractor:
    """Classe per generare frame utilizzando un modello di generazione."""
    
    def __init__(self, model_handler: BaseFrameExtractorHandler):
        self.model_handler = model_handler

    def get_frame(self, text: str) -> tuple:
        """Genera il frame dall'input testuale."""
        output_text = self.model_handler.generate_output(text)
        parsed_output = FrameParser.parse_output(output_text)

        intent = parsed_output['intent']
        argument = parsed_output['argument']
        dialogue_act = parsed_output['dialogue_act']

        frame = {}
        if parsed_output['slot_names'] and parsed_output['slot_values']:
            for i, slot_name in enumerate(parsed_output['slot_names']):
                frame[slot_name] = parsed_output['slot_values'][i]

        return intent, argument, dialogue_act, frame

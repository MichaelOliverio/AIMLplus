import sys

src_paths = [
    'c:\\...\\AIMLplus\\',
]

for src_path in src_paths:
    if src_path not in sys.path:
        sys.path.append(src_path)

import xml.etree.ElementTree as ET
from common.frame_extraction.frame_extractor import FrameExtractor

class NLU:
    def __init__(self, frame_extractor: FrameExtractor):
        self.frame_extractor = frame_extractor

    def extraction(self, user_input: str) -> None:
        """
        Processo di comprensione del linguaggio naturale (NLU). Manipola e analizza l'utterance dell'utente.
        """
        intent, argument, dialogue_act, frame = self.frame_extractor.get_frame(user_input) # estrazione frame

        return {
            'user_input': user_input,
            'intent': intent,
            'argument': argument,
            'dialogue_act': dialogue_act,
            'frame': frame
        }
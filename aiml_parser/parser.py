import sys

src_paths = [
    'c:\\...\\AIMLplus\\',
]

for src_path in src_paths:
    if src_path not in sys.path:
        sys.path.append(src_path)

import xml.etree.ElementTree as ET
from aiml_parser.category import Category

class AIMLParser:
    def __init__(self):
        self.categories = []
        self.global_slots = set()

    def load_from_aiml(self, filepath: str) -> None:
        """
        Carica le categorie da un file AIML e le inserisce nella lista delle categorie.
        """
        tree = ET.parse(filepath)
        root = tree.getroot()

        if len(self.categories) == 0:
            category_id = 0
        else:
            category_id = self.categories[-1].id + 1

        # Itera su tutti i nodi <category> dell'AIML
        for category_element in root.findall('category'):
            argument = category_element.get('argument')
            intent = category_element.get('intent')

            # Estrae il template
            template_element = category_element.find('template')
            template = ''.join(template_element.itertext()).strip() if template_element is not None else ""

            # Estrae gli atti dialogici
            dialogue_acts_list = []
            acts_element = category_element.find('acts')
            if acts_element is not None:
                for act in acts_element.findall('act'):
                    dialogue_acts_list.append(act.text)

            # Estrae il frame
            frame = {}
            correctedFrame = {}
            frame_element = category_element.find('frame')
            if frame_element is not None:
                for slot in frame_element.findall('slot'):
                    slot_name = slot.get('name')
                    slot_value = slot.get('value')
                    slot_corrected_value = slot.get('correctedValue')

                    if slot_name and slot_value:
                        if not slot_corrected_value:
                            slot_corrected_value = slot_value
                        frame[slot_name] = (slot_value)
                        correctedFrame[slot_name] = (slot_corrected_value)
                    elif slot_name:
                        # Se lo slot contiene slot-value
                        slot_value = slot.findall('slot-value')
                        if slot_value:
                            frame[slot_name] = []
                            correctedFrame[slot_name] = []
                            for value in slot_value:
                                corrected_value = value.get('correctedValue')
                                if not corrected_value:
                                    corrected_value = value.get('value')
                                frame[slot_name].append(value.get('value'))
                                correctedFrame[slot_name].append(corrected_value)

                        slot_values = slot.findall('slot-values')
                        for slot in slot_values:
                            slot_value = slot.get('value')
                            slot_corrected_value = slot.get('correctedValue')

                            if slot_name and slot_value:
                                if not slot_corrected_value:
                                    slot_corrected_value = slot_value
                                frame[slot_name] = slot_value
                                correctedFrame[slot_name] = slot_corrected_value
                            elif slot_name:
                                # Se lo slot contiene slot-value
                                slot_values = slot.findall('slot-value')
                                if slot_values:
                                    frame[slot_name] = []
                                    correctedFrame[slot_name] = []
                                    for slot_value in slot_values:
                                        slot_corrected_value = slot_value.get('correctedValue')
                                        if not slot_corrected_value:
                                            slot_corrected_value = slot_value.get('value')
                                        frame[slot_name].append(slot_value.get('value'))
                                        correctedFrame[slot_name].append(slot_corrected_value)

            # Crea la nuova categoria
            category = Category(
                id=category_id,
                intent=intent,
                argument=argument,
                dialogue_acts_list=dialogue_acts_list,
                frame=frame,
                correctedFrame=correctedFrame,
                template=template,
            )

            print(f"{category}\n")

            self.categories.append(category)
            category_id += 1

    def load_from_folder(self, folderpath: str) -> None:
        """
        Carica le categorie da una cartella contenente più file AIML e le inserisce nella lista delle categorie.
        """
        import os
        for filename in os.listdir(folderpath):
            if filename.endswith(".aiml"):
                self.load_from_aiml(os.path.join(folderpath, filename))
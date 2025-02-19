import re
import ast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameParser:
    """Parser per analizzare l'output del modello."""
    
    @staticmethod
    def parse_output(output_text: str) -> dict:
        """Analizza l'output e restituisce un dizionario con i valori estratti."""
        patterns = {
            'intent': r"'intent':\s*'([^']+)'",
            'argument': r"'argument':\s*'([^']+)'",
            'dialogue_act': r"'dialogue_act':\s*'([^']+)'",
            'slot_names': r"'slot_names':\s*\[([^\]]+)\]",
            'slot_values': r"'slot_values':\s*(\[[^\]]*\].*)"
        }

        result = {}

        for key, pattern in patterns.items():
            match = re.search(pattern, output_text)
            if match:
                if key == 'slot_values':
                    list_str = match.group(1).strip()
                    if list_str.startswith('[') and list_str.endswith(']'):
                        list_str = list_str.replace("'", '"')
                        try:
                            result[key] = eval(list_str)
                        except Exception as e:
                            logger.error(f"Errore nell'analizzare slot_values: {e}")
                            result[key] = []
                    else:
                        logger.warning("Formato errato per slot_values.")
                        result[key] = []
                elif key == 'slot_names':
                    list_str = match.group(1).strip().replace("'", '"')
                    result[key] = ast.literal_eval(f"[{list_str}]")
                else:
                    result[key] = match.group(1)
            else:
                result[key] = None

        return result

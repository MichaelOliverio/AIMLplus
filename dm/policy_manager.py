import sys

src_paths = [
    'c:\\...\\AIMLplus\\',
]

for src_path in src_paths:
    if src_path not in sys.path:
        sys.path.append(src_path)
from aiml_parser.category import Category
from typing import List, Optional, Any

# p.11 juraawsky
class PolicyManager:
    def __init__(self, categories=List[Category], method="frame", uncertainty_threshold=0):
        self.categories = categories
        self.method = method
        self.uncertainty_threshold = uncertainty_threshold
        # Our systems might have a four-tiered level of confidence with
        # - three thresholds α, β, and γ:
        # < α low confidence reject
        # ≥ α above the threshold confirm explicitly
        # ≥ β high confidence confirm implicitly
        # ≥ γ very high confidence don’t confirm at all

    def select_action(self, context, state):
        """
        Seleziona l'azione successiva basandosi sul contesto.
        """

        # todo: se la vicinanza tra farme/embedding è bassa, si potrebbe attuare
        # qualche strategia di conferma (vedi juraawsky)

        # Trova le categorie che matchano con il dialogue act corrente
        categories = self.search_categories_by_argument_intent_dialogue_acts(context, state)

        if self.method == "frame":
            return self.frame_policy(context, state, categories)
        elif self.method == "embedding":
            return self.embedding_policy(context, state, categories)
        
    def frame_policy(self, context, state, categories) -> Optional[Category]:
        """
        Frame-based policy rule-based.
        """

        current_dialogue_act = context.get("dialogue_act")

        current_frame = context.get("frame")
        frame_history = state.get("frame_history")

        # todo: se subtopic non è presente allora mi baso su quello precedente

        if current_dialogue_act == "AutoF:autoNegative":
            previous_frame = frame_history[1] if len(frame_history) > 1 else {}
            #print('previous frame: ', previous_frame)

            # merge current_input_frame with previous_local_frame
            merged_frame = previous_frame.copy()
            for slot in current_frame:
                if slot not in merged_frame:
                    merged_frame[slot] = current_frame[slot]

            # valutare inserimento del common ground

            frame = merged_frame
        else:
            frame = current_frame

        best_category, certainty_score = self.search_category_by_best_frame(categories, frame)
        
        print(f"Best category: {best_category.frame}, with score {certainty_score}")

        if certainty_score < self.uncertainty_threshold:
            return Category(
                id=None,
                intent=None,
                argument=None,
                dialogue_acts_list=None,
                frame=None,
                correctedFrame=None,
                template="Potresti essere più specifico?",
            )
        else:           
            return best_category
        
    def search_categories_by_argument_intent_dialogue_acts(self, context, state) -> List[Category]:
        """
        Cerca le categorie che corrispondono alla sequenza di atti dialogici.
        """

        categories = []
        for category in self.categories:
            if (category.intent == context.get("intent") or category.intent == None) and (category.argument == context.get("argument") or category.argument == None) and self.match_dialogue_acts(state['dialogue_act_history'], context, state):
                categories.append(category)
        return categories
    
    
    def match_dialogue_acts(self, dialogue_acts_list: List[str], context, state) -> bool:
        """
        Controlla se gli atti dialogici della categoria corrispondono alla storia recente degli atti dialogici.
        """

        dialogue_acts_list = state.get("dialogue_act_history")
        dialogue_acts_list.append(context.get("dialogue_act"))

        if len(dialogue_acts_list) > len(state['dialogue_act_history']):
            return False

        for i, act in enumerate(dialogue_acts_list):
            if state['dialogue_act_history'][i] != act:
                return False

        return True

    def search_category_by_best_frame(self, categories: List[Category], current_frame: Optional[dict] = None) -> Optional[Category]:
        """
        Cerca la categoria con il frame più simile all'input corrente.
        """
        best_category = None
        best_score = -1

        for category in categories:
            score = self.weighted_similarity(category.frame, current_frame)

            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category, best_score

    def weighted_similarity(self, frame1: dict, frame2: dict, weights: dict = {}, wildcard_weight: float = 0.2):
        """
        Calcola la similarità ponderata tra due frame, normalizzando il punteggio massimo a 1.
        
        - Una corrispondenza esatta 'chiave:valore' tra i frame contribuisce al peso pieno.
        - Una corrispondenza parziale (con valore '*') contribuisce con un peso ridotto.
        
        Parametri:
        - frame1, frame2: dizionari dei frame.
        - weights: dizionario dei pesi per ciascuna chiave.
        - wildcard_weight: peso da assegnare a corrispondenze con il jolly '*'.
        
        Restituisce:
        - La similarità normalizzata tra i due frame (valore tra 0 e 1).
        """
        total_weight = sum(weights.get(slot, 1) for slot in set(frame1.keys()).union(frame2.keys()))
        similarity = 0

        for slot in set(frame1.keys()).union(frame2.keys()):
            weight = weights.get(slot, 1)
            value1 = frame1.get(slot)
            value2 = frame2.get(slot)
            
            if value1 is not None and value2 is not None:
                if isinstance(value1, list) or isinstance(value2, list):
                    if isinstance(value1, list) and isinstance(value2, list):
                        min_len = min(len(value1), len(value2))
                        max_len = max(len(value1), len(value2))
                        
                        for i in range(min_len):
                            if value1[i] == value2[i]:
                                similarity += weight / max_len
                            elif value1[i] == "*" or value2[i] == "*":
                                similarity += (weight * wildcard_weight) / max_len
                    else:
                        similarity += (weight * wildcard_weight) / total_weight
                else:
                    if value1 == value2:
                        similarity += weight
                    elif value1 == "*" or value2 == "*":
                        similarity += weight * wildcard_weight

        return min(similarity / total_weight, 1) if total_weight else 0

    def embedding_policy(self, context, state):     
        pass
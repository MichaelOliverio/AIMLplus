# AIML+: Enhancing AIML for the Educational Domain through Frames and Large Language Models

**AIML+** is an advanced extension of **AIML (Artificial Intelligence Markup Language)** designed to enhance dialogue systems, particularly in the **educational domain**.

This project, developed in **Python**, integrates AIML with **frame**, **dialogue act** and **Large Language Models (LLMs)**, making dialogue systems more flexible, context-aware, and suitable for advanced educational applications.


## Experimental Setup

| **Parameter**                         | **Value**          |
|-------------------------------------|--------------------|
| **LoRA parameters**                 |                    |
| LoRA attention dimension            | 64                 |
| Alpha parameter                    | 16                 |
| Dropout probability                | 0.1                |
| **bitsandbytes parameters**          |                    |
| Activate 4-bit precision          | True               |
| Compute dtype for 4-bit            | float16            |
| Quantization type                 | nf4                |
| Activate nested quantization     | False              |
| **TrainingArguments parameters**     |                    |
| Number of training epochs         | 4                  |
| Enable fp16 training              | False              |
| Enable bf16 training              | True               |
| Batch size per GPU for training  | 4                  |
| Batch size per GPU for evaluation| 4                  |
| Gradient accumulation steps       | 1                  |
| Maximum gradient norm             | 0.3                |
| Initial learning rate             | 2e-4               |
| Weight decay                     | 0.001              |
| Optimizer                       | p_adamw_32bit       |
| Learning rate schedule           | cosine             |
| Warmup ratio                    | 0.03               |

*Table 1: Hyperparameters used in the experiment.*

The LLMs used in this study were fine-tuned on an A100 GPU using the Huggingface Transformers library [wolf:20]. We spent approximately 6 hours on fine-tuning and generations. The hyperparameter configuration for fine-tuning is shown in Table 1. For all the models, the same prompt was used during fine-tuning:

> `prompt: <s> [INST] Given the following input in (INPUT), you have to generate the corresponding json in (ANW) [/INST] [INPUT] ... [/INPUT] [ANW] ... [/ANW] </s>`

For the baseline, we used these four prompts, each designed to extract one of the categories (dialogue act, argument, intent, and slots):

---

### Dialogue act extraction prompt:

Given the label **dialogue act** and the following possible values:  
AutoF:autoNegative, DS:opening, SOM:initGreeting, DS:suggest, OCM:selfCorrection, SOM:initGoodbye, SOM:returnGreeting, SOM:thanking, Ta:answer, Ta:checkQuestion, Ta:propositionalQuestion, Ta:request, Ta:setQuestion, TuM:turnAccept.

You must extract the dialogue act from the user input.

Examples:

- Example 1:  
  User input: "How many transitions are there in the automaton?"  
  System response: "Ta:setQuestion"

- Example 2:  
  User input: "There are a total of 3 states: q0, q1, and q2. q0 is both initial and final state."  
  System response: "Ta:answer"

- Example 3:  
  User input: "Is q4 the final state?"  
  System response: "Ta:propositionalQuestion"

Respond only with the value, without any additional text.

---

### Argument extraction prompt:

Given the label **argument** and the following possible values:  
Automata, Language, Pattern, State, Transition, Null

You must extract the argument from the user input.

Examples:

- Example 1:  
  User input: "How many transitions are there in the automaton?"  
  System response: "Transition"

- Example 2:  
  User input: "There are a total of 3 states: q0, q1, and q2. q0 is both initial and final state."  
  System response: "State"

- Example 3:  
  User input: "What is the language of an automaton?"  
  System response: "Language"

Respond only with the value, without any additional text.

---

### Intent extraction prompt:

Given the label **intent** and the following possible values:  
fsa-theoretical, fsa-practical, Null

You must extract the intent from the user input.

Examples:

- Example 1:  
  User input: "How many transitions are there in the automaton?"  
  System response: "fsa-practical"

- Example 2:  
  User input: "There are a total of 3 states: q0, q1, and q2. q0 is both initial and final state."  
  System response: "fsa-practical"

- Example 3:  
  User input: "What is the final state of an automaton?"  
  System response: "fsa-theoretical"

Respond only with the value, without any additional text.

---

### Slots extraction prompt:

Given the following slot names:

- **alphabet:** The system or user asks or provides information about the alphabet of the automaton (e.g. `"alphabet":["1","0"]` or `"alphabet":"?"`)  
- **automatonType:** The system or user asks or provides information about the typology of the automaton (e.g. `"automatonType": "deterministic"` or `"automatonType":"?"`)  
- **finalStates:** The system or user asks or provides information about the final states of the automaton (e.g. `"finalStates": ["Q1", "Q2"]` or `"finalStates":"?"`)  
- **graphicRepresentation:** The system or user asks or provides information about the graphical representation of the automaton (e.g. `"graphicRepresentation": "pentagon"` or `"graphicRepresentation":"?"`)  
- **initialState:** The system or user asks or provides information about the initial state of the automaton (e.g. `"initialState": "Q0"` or `"initialState":"?"`)  
- **input:** The system or user asks or provides information about the input of the automaton (e.g. `"input": ["11000","1100011000"]` or `"input":"?"`)  
- **languageType:** The system or user asks or provides information about the language type of the automaton (e.g. `"languageType": "regular"` or `"languageType":"?"`)  
- **numberOfFinalStates:** The system or user asks or provides information about the number of final states of the automaton (e.g. `"numberOfFinalStates": "2"` or `"numberOfFinalStates":"?"`)  
- **numberOfStates:** The system or user asks or provides information about the number of states of the automaton (e.g. `"numberOfStates": "5"` or `"numberOfStates":"?"`)  
- **numberOfTransitions:** The system or user asks or provides information about the number of transitions of the automaton (e.g. `"numberOfTransitions": "7"` or `"numberOfTransitions":"?"`)  
- **optimalSpatialRepresentation:** The system or user asks or provides information about the optimal spatial representation of the automaton (e.g. `"optimalSpatialRepresentation":"?"`)  
- **output:** The system or user asks or provides information about the output of the automaton (e.g. `"output": "accepted/denied"` or `"output":"?"`)  
- **patternType:** The system or user asks or provides information about the pattern type of the automaton (e.g. `"patternType": "clockwise"` or `"patternType":"?"`)  
- **stateFrom:** The system or user requests or provides information about a specific starting state (e.g. `"stateFrom": "q3"` or `"stateFrom":"?"`)  
- **stateTo:** The system or user requests or provides information about a specific ending state (e.g. `"stateTo": "q3"` or `"stateTo":"?"`)  
- **stateWithMostTransitions:** The system or user asks or provides information about the state with most transitions (e.g. `"stateWithMostTransitions": "q3"` or `"stateWithMostTransitions":"?"`)  
- **stateWithoutTransitions:** The system or user asks or provides information about the states without transitions (e.g. `"stateWithoutTransitions": "q3"` or `"stateWithoutTransitions":"?"`)  
- **states:** The system or user asks or provides information about the states of the automaton (e.g. `"states": ["Q1","Q2"]` or `"states":"?"`)  
- **transitions:** The system or user asks or provides information about the transitions of the automaton (e.g. `"transitions": [["Q0","Q1","1"],["Q1","Q2","0"]]` or `"transitions":"?"`)

You must respond with a JSON containing information extracted from the user input.

Examples:

- Example 1:  
  User input: "How many transitions are there in the automaton?"  
  System response:  

  ```json
  {
    "slot names": ["numberOfTransitions"],
    "slot values": ["?"]
  }
  ```

- Example 2:
  User input: "There are a total of 3 states: q0, q1, and q2. q0 is both initial and final state."
  System response:

  ```json
  {
    "slot names": ["numberOfStates", "states", "initialState", "finalStates"],
    "slot values": ["3", ["q0", "q1", "q2"], "q0", ["q0"]]
  }
  ```

- Example 3:
  User input: "What is the final state of an automaton?"
  System response:

  ```json
  {
    "slot names": ["finalStates"],
    "slot values": ["?"]
  }
  ```

Respond only with a valid JSON, without any additional text.

---

## AIML+ Tag Description

Table 2 provides a list of AIML+ tags and their attributes, along with descriptions.

| **Tag**      | **Description**                                                                     | **Attributes**                      |
| ------------ | ----------------------------------------------------------------------------------- | ----------------------------------- |
| `aiml`       | Root element of the document.                                                       | -                                   |
| `category`   | Defines a category with eventually the argument.                                    | `argument (str)`                    |
| `acts`       | Contains a series of dialogue acts.                                                 | -                                   |
| `act`        | Defines a single dialogue act in a specific turn of the conversation.               | `time (int), value (str)`           |
| `frame`      | Defines a frame containing various elements, including slots, slot, and slot-value. | -                                   |
| `slots`      | Contains one or more slots.                                                         | -                                   |
| `slot`       | Defines a single slot.                                                              | `value (str), correctedValue (str)` |
| `slot-value` | Defines the value of a slot.                                                        | `value (str), correctedValue (str)` |
| `template`   | Defines the response.                                                               | -                                   |

*Table 2: Description of AIML+ tags and attributes*

---

## Annotations

For the annotation process described in Section $ref:annotation$, we utilized 14 dialogue act values, 7 argument labels, and 3 intent labels. Regarding slots, 19 slot names and 84 slot values were annotated. Table 3 presents the annotated labels and values:

| **Label**    | **Values**                                                                                                                                                                                                                                                                                              | **#Values** |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Dialogue act | AutoF\:autoNegative, DS\:opening, SOM\:initGreeting, DS\:suggest, OCM\:selfCorrection, SOM\:initGoodbye, SOM\:returnGreeting, SOM\:thanking, Ta\:answer, Ta\:checkQuestion, Ta\:propositionalQuestion, Ta\:request, Ta\:setQuestion, TuM\:turnAccept                                                    | 14          |
| Argument     | Automaton, Alphabet, Language, Pattern, State, Transition, Null                                                                                                                                                                                                                                         | 7           |
| Intent       | fsa-heoretical, fsa-practical, Null                                                                                                                                                                                                                                                                     | 3           |
| Slots        | alphabet, automatonType, finalStates, graphicRepresentation, initialState, input, languageType, numberOfFinalStates, numberOfStates, numberOfTransitions, optimalSpatialRepresentation, output, patternType, stateFrom, stateTo, stateWithMostTransitions, stateWithoutTransitions, states, transitions | 19          |

*Table 3: The FSA domain in our corpus*

```
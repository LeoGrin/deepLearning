Project for converting a LSTM network trained on text to a faster finite-states automaton by merging the closest LSTM memory states together (through clustering) and putting a link between 2 states if the LSTM can go from one memory state to the other in one step. 

Notebooks :

- from_LSTM_to_automata : convert a LSTM trained on a regular word database to an automaton.

- ToyLanguageExperiment and ToyLanguage : LSTM trained on the toy language formed by the well-parenthesed words from the alphabet { "(", ")", "a"}. Used to understand how the memory states of the LSTM behave, and how they should be merged. 
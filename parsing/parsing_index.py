class ParsingIndex:
    structure_index = {0: 's', 1: 'r'}
    nuclearity_index = {0: 'NS', 1: 'SN', 2: 'NN'}
    relation_index = {0: 'Joint',
                      1: 'Sequence',
                      2: 'Progression',
                      3: "Contrast",
                      4: "Supplement",
                      5: "Cause-Result",
                      6: "Result-Cause",
                      7: "Background",
                      8: "Behavior-Purpose",
                      9: "Purpose-Behavior",
                      10: "Elaboration",
                      11: "Summary",
                      12: "Evaluation",
                      13: "Statement-Illustration",
                      14: "Illustration-Statement"
                      }
    structure_dict = {'s': 0, 'r': 1}
    nuclearity_dict = {'NS': 0, 'SN': 1, 'NN': 2}
    relation_dict = {'Joint': 0,
                     'Sequence': 1,
                     'Progression': 2,
                     "Contrast": 3,
                     "Supplement": 4,
                     "Cause-Result": 5,
                     "Result-Cause": 6,
                     "Background": 7,
                     "Behavior-Purpose": 8,
                     "Purpose-Behavior": 9,
                     "Elaboration": 10,
                     "Summary": 11,
                     "Evaluation": 12,
                     "Statement-Illustration": 13,
                     "Illustration-Statement": 14
                     }

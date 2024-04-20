
import os
import evaluate
import json
from tqdm import tqdm
import numpy as np # NOTE: you don't have to use it but you are allowed to
import pandas as pd
from itertools import combinations
from tqdm import tqdm

def load_json(filename):
    """Helper to load JSON files."""
    with open(filename, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)
    return data

def save_json(mydictlist, filename):
    """Helper to save JSON files."""
    f = open(filename, 'w', encoding='utf-8')
    json.dump(mydictlist, f, ensure_ascii=False, indent=4) 
    f.close()

def create_entryid2score(entry_ids, scores):
    """Zips entry IDs and scores and creates a dictionary out of this mapping.

    Args:
        entry_ids (str list): list of data entry IDs
        scores (float list): list of scores

    Returns:
        dict: given a list of aligned entry IDs and scores creates a dictionary 
                that maps from an entry ID to the corresponding score

    """
    score_dict = {}
    for entry_id, res in zip(entry_ids, scores):
        score_dict[str(entry_id)] = res
    return score_dict

def calculate_metrics():
    ############################################################################
    # 1) Load data
    ############################################################################
    wmt_da_df = pd.read_csv(os.path.join("data", "wmt-da-human-evaluation_filtered.csv"))
    
    ############################################################################
    # 2) Load HF metrics
    ############################################################################
    # TODO: load the BLEU, BERTScore and COMET metrics from the evaluate package
    ############################################################################
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    comet = evaluate.load("comet")
    
    # NOTE: of the form {metric_name: {entry_id: score, ...}, ...}
    metric_dict = {}
        
    ############################################################################
    # 3.1) Calculate BLEU    
    ############################################################################
    # TODO: calculate the following bleu scores for each hypothesis
    #       - BLEU
    #       - BLEU-1
    #       - BLEU-4
    # Make sure to populate the metric_dict dictionary for each of these scores:
    #   For example, for BLEU-1 the metric_dict entry should look like:
    #   metric_dict["bleu-1"] = { "23423": 0.5 } where "23423" is an entry_id 
    #   and 0.5 is the BLEU-1 score it got. 
    # 
    # Feel free to use the `create_entryid2score` helper function.
    ############################################################################
    print("-" * 50)
    print("Calculating BLEU...")
    metric_dict["bleu"] = {}
    metric_dict["bleu-1"] = {}
    metric_dict["bleu-4"] = {}
    
    bleu_scores_list = []
    bleu_1_scores_list = []
    bleu_4_scores_list = []
    
    for index, row in tqdm(wmt_da_df.iterrows(), total=wmt_da_df.shape[0]):
        reference = [row['ref']]
        hypothesis = row['mt']
        bleu_result = bleu.compute(predictions=[hypothesis], references=[reference])
        bleu_scores_list.append(bleu_result['bleu'])
        bleu_1_scores_list.append(bleu_result['precisions'][0])
        bleu_4_scores_list.append(bleu_result['precisions'][3])
    
    metric_dict["bleu"] = create_entryid2score(wmt_da_df['entry_id'], bleu_scores_list)
    metric_dict["bleu-1"] = create_entryid2score(wmt_da_df['entry_id'], bleu_1_scores_list)
    metric_dict["bleu-4"] = create_entryid2score(wmt_da_df['entry_id'], bleu_4_scores_list)
    
    print("Done.")
        
    ############################################################################
    # 3.2) Calculate BERTScore
    ############################################################################
    # TODO: calculate the following BERTScore-s for each hypothesis
    #       - Precision
    #       - Recall
    #       - F-1
    # Make sure to populate the metric_dict dictionary for each of these scores.
    # Feel free to use the `create_entryid2score` helper function.
    #
    # For BERTScore, you will require to pass a `lang` parameter. Please read 
    # the documentation to figure out what that might mean. 
    # (Hint: For `lang`, you may want to use the `groupby` function of pandas dataframes!)
    ############################################################################
    print("-" * 50)
    print("Calculating BERTScore...")
    metric_dict["bertscore-precision"] = {}
    metric_dict["bertscore-recall"] = {}
    metric_dict["bertscore-f1"] = {}

    for lang, group in tqdm(wmt_da_df.groupby('lp')):
        predictions = group['mt'].tolist()
        references = group['ref'].tolist()
        lang_code = lang.split('-')[1] 
        scores = bertscore.compute(predictions=predictions, references=references, lang=lang_code)
        
        metric_dict['bertscore-precision'].update(create_entryid2score(group['entry_id'], scores['precision']))
        metric_dict['bertscore-recall'].update(create_entryid2score(group['entry_id'], scores['recall']))
        metric_dict['bertscore-f1'].update(create_entryid2score(group['entry_id'], scores['f1']))
    
    print("Done.")
    
    ############################################################################
    # 3.3) Calculate COMET
    ############################################################################
    # TODO: calculate the COMET score for each hypothesis
    # Make sure to populate the metric_dict dictionary for COMET.
    # Feel free to use the `create_entryid2score` helper function.
    ############################################################################
    print("-" * 50)
    print("Calculating COMET...")
    metric_dict["comet"] = {}

    comet_scores = comet.compute(
        predictions=wmt_da_df['mt'].tolist(), 
        references=wmt_da_df['ref'].tolist(), 
        sources=wmt_da_df['src'].tolist()
    )

    metric_dict['comet'] = create_entryid2score(wmt_da_df['entry_id'], comet_scores['scores'])
    
    
    print("Done.")
    
    ############################################################################
    # 4) Save the output in a JSON file
    ############################################################################
    save_json(metric_dict, "part3_metrics.json")
    return metric_dict
    

def evaluate_metrics():
    ############################################################################
    # 1) Load data
    ############################################################################
    wmt_da_df = pd.read_csv(os.path.join("data", "wmt-da-human-evaluation_filtered.csv"))
    print(wmt_da_df.head())
    print(len(wmt_da_df))
    
    ############################################################################
    # 2) Create ranked data for Kendall's Tau
    ############################################################################
    # TODO: For each (source, lp) group, rank the entry_id s by the "score".
    #       And then create rank_pairs_list: a list of ranking pairs which are
    #       (worse hypothesis id, better_hypothesis id)
    #       Hint: use combinations from itertools!
    ############################################################################
    rank_pairs_list = []
    grouped = wmt_da_df.groupby(['src', 'lp'])

    for name, group in grouped:
        sorted_group = group.sort_values('score', ascending=True)
        entry_ids = sorted_group['entry_id'].tolist()
        
        group_combinations = list(combinations(entry_ids, 2))
        rank_pairs_list.extend(group_combinations)

    # NOTE: The following should be ~3351
    print("Size of rank combinations: ", len(rank_pairs_list))
  
    ############################################################################
    # 2) Create a class to calculate Kendalls Tau for each metric
    ############################################################################
    # TODO: Complete the class such that each call to the class can update the 
    #       count of concordant and discordant values
    ############################################################################
    class KendallsTau:
        """
        A class to accumulate concordant and discordant instances and to
        compute Kendall's Tau correlation coefficient. 
        Helps when iteratively doing the computation.
        Feel free to implement it otherwise if you don't want to do it iteratively.
        """
        def __init__(self):
            self.concordant = 0.0
            self.discordant = 0.0
            self.total = 0.0

        def update(self, worse_hyp_score, better_hyp_score):
            """Updates the concordant and discordant values.

            Args:
                worse_hyp_score (float): the score for the worse hypothesis 
                        according to human ranking
                better_hyp_score (float): the score for the better hypothesis 
                        according to human ranking
            """
            if better_hyp_score > worse_hyp_score:
                self.concordant += 1
            elif better_hyp_score < worse_hyp_score:
                self.discordant += 1
            self.total += 1

        def compute(self):
            """
            Calculates the Kendall's Tau correlation coefficient. 
            Call when all ranked pairs have been evaluated.
            """
            num_pairs = self.total
            if num_pairs == 0:
                return 0 
            return (self.concordant - self.discordant) / num_pairs

    
    ############################################################################
    # 3) Calculate Kendall's Tau correlation for each metric
    ############################################################################
    # TODO: Populate the metrics2kendalls dictionary s.t. we get 
    #       metric2kendalls = {metric_name: correlation, ...} for all metrics
    ############################################################################
    metric_dict = load_json("part3_metrics.json")
    metric2kendalls = {}
    
    for metric_name, scores_dict in metric_dict.items():
        kendalls_tau_calculator = KendallsTau()
        
        for worse_id, better_id in rank_pairs_list:
            worse_score = scores_dict[str(worse_id)]
            better_score = scores_dict[str(better_id)]
            
            kendalls_tau_calculator.update(worse_score, better_score)

        metric2kendalls[metric_name] = kendalls_tau_calculator.compute()


    ############################################################################
    # 4) Save the output in a JSON file
    ############################################################################
    save_json(metric2kendalls, "part3_corr.json")
        

if __name__ == '__main__':
    already_predicted_scores = False
    if not already_predicted_scores:
        calculate_metrics()
    evaluate_metrics()

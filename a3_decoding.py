import torch
from typing import Any, Dict
from a3_utils import *


class GreedySearchDecoderForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)
    
    @torch.no_grad()
    def search(
        self,
        inputs: dict,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using greedy decoding. 
        This means that we always pick the next token with the highest score/probability.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping (i.e. if the next token is an EOS 
            (end-of-sentence) token, you should stop decoding) or stops at 
            max_new_tokens.
        - It only handles inputs of batch size = 1.

        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: greedy decoded best sequence made of token ids of size (1,generated_seq_len)
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # Hint (#1): There are 2 ways to pass the inputs to the model. Please open the
        # [GPT2LMHeadModel documentation](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel) 
        # and read the `Parameters` and `Returns` sections while looking at these hints.
        # Either of these approaches is fine since we don't expect the most efficient solution:
        #
        #   1. **If recomputing the past decoder processes at each decoding step:**
        #       Since the tokenizer's output dictionary keys matches these `Parameter` 
        #       i.e. arguments of the model you can directly do:
        #       ```python
        #       >> self.model(**inputs)
        #       ```
        #       Just be careful and think about how you modify the "input_ids" 
        #       and "attention_mask" keys across decoding steps. 
        
        #   2. **If using cached decoder hidden states at each decoding step:**
        #       To speed up the process (although *not required*) you can also get 
        #       the computed key/values hidden-states *so far* with `use_cache=True`
        #       where in the first step you may need to do:
        #       ```python
        #       >> self.model(**inputs, use_cache=True)
        #       ```
        #       This will return an extra dictionary entry called "past_key_values".
        #       In the next steps you would do, assuming your previous output 
        #       dict is called `outputs`:
        #       ```python
        #       >> self.model(**inputs, use_cache=True, past_key_values=outputs["past_key_values"])
        #       ```
        #       Again, be careful as to how you modify the "input_ids" and 
        #       "attention_mask" keys across decoding steps. In particular the 
        #       cached setting expects them to be different from the non-cached 
        #       setting. Read the `Parameters` and `Returns` sections of the 
        #       GPT2LMHeadModel carefully.
        #
        # Hint (#2): You can implement and use the `self.prepare_next_inputs` 
        #   function in `a3_utils.py` inherited by all decoding and sampling classes 
        #   (although you are not required to) to reduce repeated code and make it
        #   more readable. There isn't a unique solution for this so use it as you wish
        #   or create another function in this super class.
        ########################################################################
 
        self.model.eval()

        # Ensure all inputs are on the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()} 

        inputs = inputs
        for _ in range(max_new_tokens):
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits,dim=-1).unsqueeze(-1)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            inputs = self.prepare_next_inputs(
                model_inputs = inputs,
                new_token_id = next_token_id,
                use_cuda=(device.type == 'cuda'))

        return inputs['input_ids']

class BeamSearchDecoderForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)
    
    @torch.no_grad()
    def search(
        self,
        inputs: dict,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences=1,
        length_penalty: float = 0.0
    ) -> dict: 
        """Generates sequences of token ids with self.model, 
        (which has a language modeling head) using beam search. This means that 
        given a probability distribution over the possible next tokens and 
        a beam width (here num_beams), needs to keep track of the most probable 
        num_beams candidates. (Hint: use log probabilities!)

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens. 
        - It only handles inputs of batch size = 1.
        - It only handles beam size > 1.
        - It includes a length_penalty variable that controls the score assigned 
            to a long generation. This is implemented by exponiating the amount 
            of newly generated tokens to this value. Then, divide the score which 
            can be calculated as the sum of the log probabilities so far.
        
        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)
            num_beams (int): number of beams for beam search
            num_return_sequences (int, optional):
                the amount of best sequences to return. Cannot be more than beam size.
                Defaults to 1.
            length_penalty (float, optional): 
                exponential penalty to the length that is used with beam-based generation. 
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. 
                Defaults to 0.0.

        Returns:
            dict: dictionary with two key values:
                    - "sequences": torch.LongTensor depicting the best generated sequences (token ID tensor) 
                        * shape (num_return_sequences, maximum_generated_sequence_length)
                        * ordered from best scoring sequence to worst
                        * if a sequence has reached end of the sentence, 
                          you can fill the rest of the tensor row with the pad token ID
                    - "sequences_scores": length penalized log probability score list, ordered by best score to worst
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(
            inputs, 
            max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################
        
        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs['input_ids'].repeat(num_beams, 1)
        attention_mask = inputs['attention_mask'].repeat(num_beams, 1)

        start = input_ids.shape[1]

        beam_scores = torch.full((num_beams,), float('-inf'), device=device)
        beam_scores[0] = 0.0

        for _ in range(max_new_tokens):

            outputs = self.model(input_ids, attention_mask=attention_mask)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            next_token_scores += beam_scores[:, None].expand_as(next_token_scores)

            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(1, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(next_token_scores, num_beams, dim=1, largest=True, sorted=True)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            finished = next_tokens == self.eos_token_id
            ongoing_tokens = next_tokens[~finished]
            ongoing_indices = next_indices[~finished]
            ongoing_scores = next_token_scores[~finished]

            _, top_idxs = torch.topk(ongoing_scores, min(num_beams, ongoing_scores.size(0)))

            beam_tokens, beam_indices, beam_scores = ongoing_tokens[top_idxs], ongoing_indices[top_idxs], ongoing_scores[top_idxs]

            input_ids = torch.cat([input_ids[beam_indices, :], beam_tokens.unsqueeze(1)], dim=-1)

            new_mask = torch.tensor([[1] for _ in range(num_beams)], device=device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)


        if length_penalty != 0:
            adjusted_scores = beam_scores / ((input_ids.size(1) - start) ** length_penalty)
        else:
            adjusted_scores = beam_scores

        sorted_indices = torch.argsort(adjusted_scores, descending=True)
        top_input_ids = input_ids[sorted_indices][:num_return_sequences]
        top_scores = adjusted_scores[sorted_indices][:num_return_sequences]

        padded_input_ids = pad_sequences(top_input_ids, self.tokenizer.eos_token_id, num_return_sequences, top_input_ids.size(1))
        input_ids, beam_scores = padded_input_ids, top_scores

        return {
            'sequences': input_ids.long()[:num_return_sequences, :],
            'sequences_scores': beam_scores[:num_return_sequences].tolist()
        }


"""
Name of the AI-based tool : ChatGPT
Prompt used if any : Could you pad sequences of token IDs to a uniform length for processing up to max_len using the EOS token.
Output of the tool : The function below 
A brief explanation of how you verified the correctness of the output for your use case and the modifications you did :
No modification, it helped me pass A3 tests
A brief explanation of why your AI-based code does what it is supposed to do :
Sequences shorter than max_len are extended using the eos_token_id, ensuring all sequences in the batch are of equal length. This uniformity is required for consistent processing.
A mask is created to identify positions in the tensor where the original input_ids need to be placed, and the padding (EOS tokens) should fill the remaining positions. 
"""
def pad_sequences(input_ids, eos_token_id, num_return_sequences, max_len):
    """
    Pads the input sequences up to max_len using the EOS token.

    Args:
        input_ids (torch.Tensor): The tensor containing the sequences to be padded.
        eos_token_id (int): The end of sequence token id used for padding.
        num_return_sequences (int): The number of sequences expected to return.
        max_len (int): The maximum length to pad the sequences to.

    Returns:
        torch.Tensor: The padded tensor of input sequences.
    """
    padded_input_ids = torch.full((num_return_sequences, max_len), eos_token_id, dtype=torch.long, device=input_ids.device)
    mask = torch.arange(max_len, device=input_ids.device).expand(num_return_sequences, max_len) < input_ids.size(1)
    padded_input_ids[mask] = input_ids.flatten()

    return padded_input_ids

def main():
    ############################################################################
    # NOTE: You can use this space for testing but you are not required to do so!
    ############################################################################
    seed = 421
    torch.manual_seed(seed)
    torch.set_printoptions(precision=16)
    model_name = "vicgalle/gpt2-alpaca-gpt4"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


if __name__ == '__main__':
    main()

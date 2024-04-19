import torch
from typing import Any, Dict
from a3_utils import *


class TopKSamplerForCausalLM(GeneratorForCausalLM):
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
    def sample(
        self,
        inputs: dict,
        top_k: int,
        temperature: float,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-k sampling. 
        This means that we sample the next token from the top-k scoring tokens 
        by using their probability values.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - It only handles top_k => 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.
        
        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_k (int): the number of highest probability vocabulary tokens 
                         to keep for top-k filtering/sampling
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-k sampled sequence made of token ids of size (1,generated_seq_len)
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens, top_k=top_k)
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

        generated_tokens = inputs['input_ids']

        for _ in range(max_new_tokens):
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            scaled_logits = logits / temperature

            log_probabilities = torch.nn.functional.log_softmax(scaled_logits, dim=-1)
            top_k_log_probs, top_k_indices = torch.topk(log_probabilities, top_k)

            probabilities = torch.exp(top_k_log_probs)
            next_token_id = torch.multinomial(probabilities, 1) 

            next_token_id = top_k_indices.gather(-1, next_token_id) 

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            generated_tokens = torch.cat((generated_tokens, next_token_id), dim=1)
            
            inputs = self.prepare_next_inputs(
                model_inputs=inputs,
                new_token_id=next_token_id,
                use_cuda=(device.type == 'cuda'))

        return generated_tokens




class TopPSamplerForCausalLM(GeneratorForCausalLM):
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
    def sample(
        self,
        inputs: dict,
        top_p: float,
        temperature: float,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-p sampling. 
        This means that we sample the next token from the smallest set of most 
        probable tokens with probabilities that cumulatively add up to top_p *or higher*.
        If there are no tokens falling in the top_p cumulative probability mass 
        (e.g. because the top scoring tokens probability is larger than top_p) 
        then samples the top scoring token.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.

        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_p (float): the cumulative probability mass to select the smallest 
                           set of most probable tokens with probabilities that 
                           cumulatively add up to top_p or higher.
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-p sampled sequence made of token ids of size (1,generated_seq_len)
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
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_tokens = inputs['input_ids']

        for _ in range(max_new_tokens):
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(torch.nn.functional.log_softmax(sorted_logits, dim=-1)), dim=-1)

            # Remove tokens with cumulative probability above the threshold top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('Inf')
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probabilities = torch.exp(log_probs)  # Convert log probabilities back for sampling

            next_token_id = torch.multinomial(probabilities, 1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            generated_tokens = torch.cat((generated_tokens, next_token_id), dim=1)
            inputs = self.prepare_next_inputs(
                model_inputs=inputs,
                new_token_id=next_token_id,
                use_cuda=(device.type == 'cuda'))

        return generated_tokens


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

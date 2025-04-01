"""
LLM Model class for loading and inference.
"""
import os
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class LLMModel:
    """Class for loading and using LLMs."""
    
    def __init__(
        self, 
        model_name_or_path: str,
        device: Optional[str] = None,
        use_half_precision: bool = True,
        max_memory: Optional[Dict[int, str]] = None,
        mock_mode: bool = False  # Added mock mode for demonstration
    ):
        """
        Initialize the LLM model.
        
        Args:
            model_name_or_path: HF model name or path to local model
            device: Device to use (cuda, cpu, etc.)
            use_half_precision: Whether to use half precision
            max_memory: Memory configuration for loading large models
            mock_mode: If True, don't actually load models (for demo purposes)
        """
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half_precision = use_half_precision and self.device == "cuda"
        self.max_memory = max_memory
        self.mock_mode = mock_mode
        
        self.tokenizer = None
        self.model = None
        self.stopping_criteria = None
        
        if not self.mock_mode:
            self.tokenizer = self._load_tokenizer()
            self.model = self._load_model()
            
            # Define stopping criteria
            stop_token_ids = [self.tokenizer.eos_token_id] if self.tokenizer else [0]
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        try:
            return AutoTokenizer.from_pretrained(self.model_name_or_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("This is expected if you don't have access to the model or aren't authenticated with Hugging Face.")
            return None
    
    def _load_model(self) -> PreTrainedModel:
        """Load the model."""
        try:
            # Parameters for loading the model
            kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16 if self.use_half_precision else torch.float32,
            }
            
            if self.max_memory is not None:
                kwargs["max_memory"] = self.max_memory
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **kwargs
            )
            
            # Make sure the model is in the desired state
            if self.device != "cuda" or not self.use_half_precision:
                model = model.to(self.device)
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("This is expected if you don't have access to the model or aren't authenticated with Hugging Face.")
            return None
    
    def generate(
        self, 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated text
        """
        if self.mock_mode or self.model is None:
            print("Running in mock mode or model not loaded. Returning demo response.")
            return f"[DEMO RESPONSE] This is a simulated response to: '{prompt}'\nIn a real setup with proper authentication and hardware, the model would generate an actual response."
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and return the generated text, skipping the input prompt
        input_length = inputs.input_ids.shape[1]
        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


if __name__ == "__main__":
    # Example usage
    print("Creating model in mock mode for demonstration")
    model = LLMModel("mistralai/Mistral-7B-Instruct-v0.2", mock_mode=True)
    response = model.generate("What is the capital of France?")
    print(response)


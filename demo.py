#!/usr/bin/env python
"""
LLM Playground Demonstration Script

This script demonstrates the basic functionality of the LLM Playground
without requiring access to actual models or authentication.
"""

import os
import sys
import time

# Make sure we can import from the src directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # Import required packages
    import torch
    import transformers
    
    # Import our model class
    from src.inference.model import LLMModel
    
    print("-" * 50)
    print("LLM Playground Demonstration")
    print("-" * 50)
    
    # Print environment info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 50)
    
    # Create a model in mock mode
    print("Creating LLM in mock mode...")
    model = LLMModel(
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
        mock_mode=True
    )
    print("Model created successfully in mock mode")
    print("-" * 50)
    
    # Generate some text
    prompts = [
        "What is machine learning?",
        "Write a short poem about artificial intelligence.",
        "Explain the concept of transfer learning."
    ]
    
    print("Generating responses to prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        
        start_time = time.time()
        response = model.generate(prompt, max_new_tokens=100)
        end_time = time.time()
        
        print(f"Response (generated in {end_time - start_time:.2f} seconds):")
        print(response)
        print("-" * 30)
    
    print("\nDemonstration completed successfully!")
    print("""
    To use real models:
    1. Install dependencies: pip install -r requirements.txt
    2. Authenticate with Hugging Face: huggingface-cli login
    3. Set mock_mode=False when creating the model
    """)
    
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install the required packages: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)


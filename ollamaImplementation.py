import os
import json
import re
from typing import Tuple, List, Dict
import requests

# Import existing functions
from singleCharac import (
    text_to_bitstream, 
    bitstream_to_text,
    select_entity, 
    get_all_entities_from_data,
    decode_bitstream
)

from chunking_module import (
    calculate_max_chunk_size,
    encode_chunks_with_metadata,
    reconstruct_bitstream
)

# ==============================================================================
# --- OLLAMA API WRAPPER ---
# ==============================================================================

class OllamaAgent:
    """Wrapper for Ollama API calls - no rate limiting needed!"""
    
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama agent.
        
        Args:
            model_name: Name of the Ollama model (e.g., 'mistral', 'tinyllama', 'llama2')
            base_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        
        # Verify Ollama is running and model is available
        self._verify_setup()
    
    def _verify_setup(self):
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'].split(':')[0] for m in models]
                if self.model_name not in model_names:
                    print(f"⚠️  Warning: Model '{self.model_name}' not found.")
                    print(f"Available models: {model_names}")
                    print(f"Run: ollama pull {self.model_name}")
                else:
                    print(f"✅ Ollama ready with model: {self.model_name}")
            else:
                print("⚠️  Warning: Could not connect to Ollama API")
        except Exception as e:
            print(f"⚠️  Warning: Ollama connection error: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate text using Ollama - NO RATE LIMITING!
        
        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "options": {
                    "num_predict": 150  # Max tokens
                }
            }
            
            response = requests.post(self.api_endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: Ollama API returned status {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"


# ==============================================================================
# --- GENERATION WITH OLLAMA (NO RATE LIMITING!) ---
# ==============================================================================

def generate_conversation_ollama(agent: OllamaAgent, entity_name: str, 
                                entity_description: str) -> str:
    """
    Generate a sentence containing the target entity using Ollama.
    NO RATE LIMITING - call as many times as you need!
    """
    print(f"--- Generating sentence for: {entity_name} ---")
    
    prompt = f"""You are a Generation Agent (GA). Your task is to generate a single, fluent sentence that contains a specific entity.

**Target Entity:** "{entity_name}"
**Additional Information:** {entity_description}

**CRITICAL REQUIREMENTS:**
1. **Must Contain Entity:** The sentence you generate MUST contain the exact phrase "{entity_name}".
2. **Be Natural:** The sentence should be grammatically correct and sound natural.

**Example:**
- **Target Entity:** "Las Vegas"
- **Generated Sentence:** "Las Vegas is a popular tourist destination known for its vibrant nightlife, world-class entertainment, and bustling casinos."

Now, generate a sentence containing "{entity_name}".

**SENTENCE:**"""
    
    generated_text = agent.generate(prompt)
    
    # Clean up response
    if generated_text.startswith("**SENTENCE:**"):
        generated_text = generated_text.replace("**SENTENCE:**", "").strip()
    
    return generated_text


def check_conversation_quality_ollama(agent: OllamaAgent, conversation: str, 
                                     entity_name: str, entity_description: str) -> dict:
    """
    Check Agent: Verifies sentence quality using Ollama.
    NO RATE LIMITING!
    """
    print(f"\n--- Running Check Agent ---")
    
    prompt = f"""You are a meticulous Check Agent (CA). Your job is to evaluate a sentence based on strict entity control rules.

**Target Entity:** "{entity_name}"
**Entity Categories to Scan For:** PERSON, LOCATION, ORGANIZATION, TIME, EVENT

**Sentence to Evaluate:**
"{conversation}"

---
**EVALUATION CHECKS:**

1. **Target Presence:**
   - **Question:** Does the sentence contain the exact target entity "{entity_name}"? If not, this is a failure.

2. **Unwanted Entity Exclusion:**
   - **Question:** Besides "{entity_name}", does the sentence contain ANY OTHER recognizable named entities from the categories (PERSON, LOCATION, ORGANIZATION, TIME, EVENT)?
   - For example, if the target is "Mr. Kee" (a PERSON), the presence of "children" (another PERSON) or "farm" (a LOCATION) is a failure.

---
**YOUR TASK:**
Based on your evaluation, provide a JSON response.

**Response Format (JSON only, no other text):**
{{
  "target_present": true or false,
  "unwanted_entities_found": ["List any other named entities found, or an empty list if none."],
  "is_compliant": true or false,
  "recommendation": "ACCEPT" or "REGENERATE",
  "reason": "Briefly explain why."
}}"""
    
    response_text = agent.generate(prompt, temperature=0.3)  # Lower temp for structured output
    
    try:
        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        print(f"Check Agent Compliant: {result.get('is_compliant')}")
        print(f"Recommendation: {result.get('recommendation', 'UNKNOWN')}")
        print(f"Reason: {result.get('reason', 'No reason provided')}")
        
        return result
    except Exception as e:
        print(f"Check Agent parsing error: {e}")
        return {
            "recommendation": "REGENERATE", 
            "reason": "Check agent failed to parse.", 
            "is_compliant": False
        }


def regenerate_with_feedback_ollama(agent: OllamaAgent, entity_name: str, 
                                   entity_description: str, previous_conversation: str, 
                                   feedback: dict, attempt: int) -> str:
    """
    Regenerate sentence with feedback using Ollama.
    NO RATE LIMITING!
    """
    print(f"\n--- Regenerating (Attempt {attempt}) ---")
    
    reason = feedback.get('reason', 'The sentence did not meet the requirements.')
    
    prompt = f"""You are a Generation Agent (GA). Your previous attempt to generate a sentence was incorrect. You must now generate an improved version based on the feedback.

**Target Entity:** "{entity_name}"

**Previous Attempt (FAILED):**
"{previous_conversation}"

**Reason for Failure (from Check Agent):**
{reason}

**INSTRUCTIONS FOR IMPROVEMENT:**
1. Your new sentence MUST contain the target entity "{entity_name}".
2. Your new sentence MUST NOT contain the other entities that were flagged in the failure reason.
3. Generate only a single, natural-sounding sentence.

**Example of a good regeneration:**
- **Target:** "Las Vegas"
- **Previous Attempt:** "It is a popular tourist destination known for its vibrant nightlife, world-class entertainment, and bustling casinos"
- **Feedback:** "Please regenerate a fluent sentence with "Las Vegas", do not generate other LOCATION-related entities"
- **New Sentence:** "Las Vegas is a popular tourist destination known for its vibrant nightlife, world-class entertainment, and bustling casinos."

Now, generate an IMPROVED sentence for "{entity_name}" that fixes the specified problem.

**IMPROVED SENTENCE:**"""
    
    generated_text = agent.generate(prompt)
    
    # Clean up response
    if generated_text.startswith("**IMPROVED SENTENCE:**"):
        generated_text = generated_text.replace("**IMPROVED SENTENCE:**", "").strip()
    
    return generated_text


def generate_validated_sentence_ollama(agent: OllamaAgent, entity_name: str, 
                                      entity_desc: str, max_attempts: int = 3) -> str:
    """
    Generate and validate sentence using Ollama.
    NO SLEEP/DELAYS NEEDED - No rate limiting with local Ollama!
    """
    stego_conversation = ""
    check_result = {}
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n[Attempt {attempt}/{max_attempts}] Generating for '{entity_name}'...")
        
        if attempt == 1:
            stego_conversation = generate_conversation_ollama(agent, entity_name, entity_desc)
        else:
            stego_conversation = regenerate_with_feedback_ollama(
                agent, entity_name, entity_desc, stego_conversation, check_result, attempt
            )
        
        check_result = check_conversation_quality_ollama(agent, stego_conversation, 
                                                        entity_name, entity_desc)
        
        if check_result.get('is_compliant'):
            print(f"✅ Sentence accepted for '{entity_name}'")
            return stego_conversation
        elif attempt < max_attempts:
            print(f"⚠️ Attempt {attempt} failed, regenerating...")
    
    print(f"⚠️ Max attempts reached for '{entity_name}', using last version")
    return stego_conversation


# ==============================================================================
# --- EXTRACTION WITH OLLAMA ---
# ==============================================================================

def extract_entity_ollama(agent: OllamaAgent, stego_text: str, 
                         entity_list: list) -> str:
    """
    Extract entity from stego text using Ollama.
    First tries direct pattern matching, then uses LLM if needed.
    """
    print("\n--- Running Extraction Agent ---")
    
    # First try: Direct pattern matching (fastest, most reliable)
    for entity in entity_list:
        pattern = r'\b' + re.escape(entity) + r'\b'
        if re.search(pattern, stego_text):
            print(f"✓ Valid entity extracted (Direct Match): {entity}")
            return entity
    
    # Second try: Case-insensitive search
    print("⚠️ No direct match found. Trying case-insensitive search...")
    for entity in entity_list:
        pattern = r'\b' + re.escape(entity) + r'\b'
        if re.search(pattern, stego_text, re.IGNORECASE):
            print(f"✓ Valid entity extracted (Case-Insensitive Match): {entity}")
            return entity
    
    # Third try: Use LLM for extraction
    print("⚠️ Using LLM for entity extraction...")
    
    # Limit entity list for prompt (avoid token limits)
    sample_entities = entity_list[:50] if len(entity_list) > 50 else entity_list
    
    prompt = f"""Extract the main entity from this sentence. Choose from the following list:

{', '.join(sample_entities[:20])}
... and {len(entity_list) - 20} more entities.

Sentence: "{stego_text}"

Respond with ONLY the entity name, nothing else."""
    
    extracted = agent.generate(prompt, temperature=0.1).strip()
    
    # Verify extracted entity is in the list
    for entity in entity_list:
        if entity.lower() in extracted.lower():
            print(f"✓ Valid entity extracted (LLM): {entity}")
            return entity
    
    print(f"✗ CRITICAL ERROR: Could not find any known entity in the sentence: '{stego_text}'")
    return "Error: No valid entity found in the final sentence."


# ==============================================================================
# --- MAIN WORKFLOW WITH OLLAMA ---
# ==============================================================================

def main_ollama_chunked_encoding_decoding(model_name: str = "mistral"):
    """
    Main function using Ollama for chunked encoding and decoding.
    NO RATE LIMITING!
    
    Args:
        model_name: Ollama model to use ('mistral', 'tinyllama', 'llama2', etc.)
    """
    # Initialize Ollama agent
    agent = OllamaAgent(model_name=model_name)
    
    # Load ontology
    ontology_file = 'ontology_with_probabilities.json'
    try:
        with open(ontology_file, 'r') as f:
            ontology_data = json.load(f)
        all_entity_names = get_all_entities_from_data(ontology_data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FATAL ERROR: Could not load '{ontology_file}'. Details: {e}")
        exit()
    
    # ========================================
    # ENCODING PHASE
    # ========================================
    
    secret_message = "Hello!"
    
    print("="*70)
    print("CHUNKED ENCODING WORKFLOW (OLLAMA - NO RATE LIMITS!)")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Secret Message: '{secret_message}'")
    print(f"Message Length: {len(secret_message)} characters")
    print("="*70)
    
    # Convert message to bitstream
    secret_bits = text_to_bitstream(secret_message)
    print(f"Total bitstream length: {len(secret_bits)} bits")
    
    # Calculate chunk size
    chunk_size = calculate_max_chunk_size(ontology_data)
    
    # Create chunks with metadata
    chunks, metadata = encode_chunks_with_metadata(secret_bits, chunk_size)
    
    # Encode each chunk to an entity
    selected_entities = []
    print(f"\n{'='*60}")
    print("ENCODING CHUNKS TO ENTITIES")
    print(f"{'='*60}")
    
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx+1}/{len(chunks)} ---")
        print(f"Bitstream: {chunk}")
        
        entity_node, bit_length = select_entity(ontology_data, chunk)
        entity_info = {
            'name': entity_node.get('name'),
            'id': entity_node.get('id'),
            'chunk_index': idx,
            'bit_length': bit_length
        }
        selected_entities.append(entity_info)
        print(f"Selected entity: {entity_info['name']}")
    
    # Generate stego sentences
    print(f"\n{'='*70}")
    print("GENERATING STEGO SENTENCES (NO DELAYS!)")
    print(f"{'='*70}")
    
    stego_sentences = []
    for entity_info in selected_entities:
        print(f"\n--- Chunk {entity_info['chunk_index']+1}/{len(selected_entities)} ---")
        print(f"Target Entity: {entity_info['name']}")
        
        sentence = generate_validated_sentence_ollama(
            agent,
            entity_info['name'], 
            "a relevant concept",
            max_attempts=3
        )
        stego_sentences.append(sentence)
        print(f"Generated: {sentence}")
    
    # Create final stego paragraph
    stego_paragraph = ' '.join(stego_sentences)
    
    print(f"\n{'='*70}")
    print("FINAL STEGO TEXT")
    print(f"{'='*70}")
    print(stego_paragraph)
    print(f"{'='*70}")
    
    # ========================================
    # DECODING PHASE
    # ========================================
    
    print(f"\n{'='*70}")
    print("CHUNKED DECODING WORKFLOW")
    print(f"{'='*70}")
    
    decoded_chunks = []
    
    for idx, sentence in enumerate(stego_sentences):
        print(f"\n--- Decoding Chunk {idx+1}/{len(stego_sentences)} ---")
        print(f"Sentence: {sentence[:100]}...")
        
        # Extract entity
        entity_name = extract_entity_ollama(agent, sentence, all_entity_names)
        
        if "Error" in entity_name:
            print(f"ERROR: Could not extract entity from chunk {idx+1}")
            decoded_chunks.append("0" * metadata['chunk_size'])
            continue
        
        print(f"Extracted entity: {entity_name}")
        
        # Decode bitstream
        decoded_bits = decode_bitstream(ontology_data, entity_name, metadata['chunk_size'])
        
        if "Error" in decoded_bits:
            print(f"ERROR: Could not decode chunk {idx+1}")
            decoded_chunks.append("0" * metadata['chunk_size'])
            continue
        
        decoded_chunks.append(decoded_bits)
        print(f"Decoded bits: {decoded_bits}")
    
    # Reconstruct original bitstream
    full_bitstream = reconstruct_bitstream(decoded_chunks, metadata)
    
    # Convert back to text
    decoded_message = bitstream_to_text(full_bitstream)
    
    # ========================================
    # VERIFICATION
    # ========================================
    
    print(f"\n{'='*70}")
    print("PIPELINE VERIFICATION REPORT")
    print(f"{'='*70}")
    print(f"Original Message:  '{secret_message}'")
    print(f"Decoded Message:   '{decoded_message}'")
    print(f"Length Original:   {len(secret_message)} chars")
    print(f"Length Decoded:    {len(decoded_message)} chars")
    print("-"*70)
    
    if secret_message == decoded_message:
        print("✅ SUCCESS: Perfect match!")
        print(f"✅ Successfully encoded/decoded {len(selected_entities)} chunks")
    else:
        print("❌ FAILURE: Messages don't match")
        print(f"\nDifferences:")
        for i, (orig, dec) in enumerate(zip(secret_message, decoded_message)):
            if orig != dec:
                print(f"  Position {i}: '{orig}' != '{dec}'")
    
    print("="*70)
    
    return {
        'original': secret_message,
        'decoded': decoded_message,
        'stego_text': stego_paragraph,
        'num_chunks': len(selected_entities),
        'metadata': metadata
    }


# ==============================================================================
# --- ENTRY POINT ---
# ==============================================================================

if __name__ == '__main__':
    print("\n🚀 OLLAMA SEMANTIC STEGANOGRAPHY")
    print("=" * 70)
    print("✅ NO RATE LIMITING!")
    print("✅ NO API COSTS!")
    print("✅ RUNS LOCALLY!")
    print("=" * 70)
    
    # Choose your model
    print("\nAvailable Ollama models:")
    print("  1. mistral (recommended - best quality)")
    print("  2. tinyllama (fastest - lower quality)")
    print("  3. llama2 (good balance)")
    print("  4. phi (Microsoft's model)")
    
    choice = input("\nSelect model (1-4) or press Enter for mistral: ").strip()
    
    model_map = {
        "1": "mistral",
        "2": "tinyllama",
        "3": "llama2",
        "4": "phi",
        "": "mistral"
    }
    
    model_name = model_map.get(choice, "mistral")
    
    print(f"\n🎯 Using model: {model_name}")
    print("\nMake sure Ollama is running: ollama serve")
    print(f"And the model is pulled: ollama pull {model_name}\n")
    
    input("Press Enter to start...")
    
    try:
        result = main_ollama_chunked_encoding_decoding(model_name=model_name)
        
        # Save results
        with open('ollama_chunked_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\n✅ Results saved to 'ollama_chunked_results.json'")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
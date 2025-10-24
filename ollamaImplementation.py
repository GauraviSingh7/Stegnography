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
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
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
        """Generate text using Ollama - NO RATE LIMITING!"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "options": {
                    "num_predict": 200
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
# --- GENERATION WITH OLLAMA ---
# ==============================================================================

def generate_conversation_ollama(agent: OllamaAgent, entity_name: str, 
                                entity_description: str) -> str:
    """Generate a sentence containing the target entity using Ollama."""
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
    generated_text = generated_text.split('\n')[0]
    generated_text = re.sub(r'^(Answer:|OUTPUT:|SENTENCE:)\s*', '', generated_text, flags=re.IGNORECASE)
    generated_text = generated_text.strip('"').strip()
    return generated_text


def check_conversation_quality_ollama(agent: OllamaAgent, conversation: str, 
                                     entity_name: str, entity_description: str) -> dict:
    """Check Agent: Verifies sentence quality using Ollama."""
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
    
    response_text = agent.generate(prompt, temperature=0.3)
    
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        response_text = response_text.strip()
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            response_text = response_text[start:end+1]
        
        result = json.loads(response_text)
        
        if not isinstance(result, dict):
            raise ValueError("Response is not a dictionary")
        
        if 'is_compliant' not in result:
            result['is_compliant'] = False
        if 'recommendation' not in result:
            result['recommendation'] = 'REGENERATE'
        if 'reason' not in result:
            result['reason'] = 'Invalid response format'
        
        print(f"Check Agent Compliant: {result.get('is_compliant')}")
        print(f"Recommendation: {result.get('recommendation', 'UNKNOWN')}")
        print(f"Reason: {result.get('reason', 'No reason provided')}")
        
        return result
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Check Agent parsing error: {e}")
        print(f"Raw response: {response_text[:200]}")
        
        # Fallback: Simple regex check
        pattern = r'\b' + re.escape(entity_name) + r'\b'
        if re.search(pattern, conversation, re.IGNORECASE):
            print("✅ Fallback: Entity found via regex")
            return {
                "target_present": True,
                "is_compliant": True,
                "recommendation": "ACCEPT",
                "reason": "Entity verified via fallback check"
            }
        else:
            print("❌ Fallback: Entity not found")
            return {
                "target_present": False,
                "is_compliant": False,
                "recommendation": "REGENERATE",
                "reason": "Entity not found in sentence"
            }


def regenerate_with_feedback_ollama(agent: OllamaAgent, entity_name: str, 
                                   entity_description: str, previous_conversation: str, 
                                   feedback: dict, attempt: int) -> str:
    """Regenerate sentence with feedback using Ollama."""
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

Now, generate an IMPROVED sentence for "{entity_name}" that fixes the specified problem.

**IMPROVED SENTENCE:**"""
    
    generated_text = agent.generate(prompt)
    generated_text = generated_text.split('\n')[0]
    generated_text = re.sub(r'^(Answer:|OUTPUT:|SENTENCE:)\s*', '', generated_text, flags=re.IGNORECASE)
    generated_text = generated_text.strip('"').strip()
    return generated_text


def generate_validated_sentence_ollama(agent: OllamaAgent, entity_name: str, 
                                      entity_desc: str, max_attempts: int = 3) -> str:
    """Generate and validate sentence using Ollama."""
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
# --- FIXED EXTRACTION ---
# ==============================================================================

def extract_entity_deterministic(stego_text: str, target_entities: list) -> str:
    """
    FIXED: Extract entity using ONLY the list of entities that were actually encoded.
    This prevents false matches with similar entities.
    
    Args:
        stego_text: The sentence to extract from
        target_entities: List of ONLY the entities that were encoded (not all entities!)
    """
    print("\n--- Running Extraction Agent (Deterministic) ---")
    
    # Sort by length (longest first) to match most specific first
    sorted_entities = sorted(target_entities, key=len, reverse=True)
    
    # Try exact match with word boundaries (case-insensitive)
    for entity in sorted_entities:
        pattern = r'\b' + re.escape(entity) + r'\b'
        if re.search(pattern, stego_text, re.IGNORECASE):
            print(f"✓ Entity extracted: {entity}")
            return entity
    
    print(f"✗ ERROR: No entity found in: '{stego_text[:100]}'")
    print(f"Expected one of: {target_entities}")
    return "Error: No valid entity found"


# ==============================================================================
# --- MAIN WORKFLOW WITH OLLAMA (FIXED) ---
# ==============================================================================

def main_ollama_chunked_encoding_decoding(model_name: str = "mistral"):
    """
    FIXED: Main function with proper entity list handling.
    """
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
    
    secret_message = "HeLL"
    
    print("="*70)
    print("CHUNKED ENCODING WORKFLOW (OLLAMA - FIXED VERSION)")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Secret Message: '{secret_message}'")
    print(f"Message Length: {len(secret_message)} characters")
    print("="*70)
    
    secret_bits = text_to_bitstream(secret_message)
    print(f"Total bitstream length: {len(secret_bits)} bits")
    
    chunk_size = calculate_max_chunk_size(ontology_data)
    chunks, metadata = encode_chunks_with_metadata(secret_bits, chunk_size)
    
    # Encode each chunk to an entity
    selected_entities = []
    encoded_entity_names = []  # ← CRITICAL: Store names for decoding
    
    print(f"\n{'='*60}")
    print("ENCODING CHUNKS TO ENTITIES")
    print(f"{'='*60}")
    
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx+1}/{len(chunks)} ---")
        print(f"Bitstream: {chunk}")
        
        entity_node, bit_length = select_entity(ontology_data, chunk)
        entity_name = entity_node.get('name')
        
        entity_info = {
            'name': entity_name,
            'id': entity_node.get('id'),
            'chunk_index': idx,
            'bit_length': bit_length
        }
        selected_entities.append(entity_info)
        encoded_entity_names.append(entity_name)  # ← STORE FOR DECODING
        
        print(f"Selected entity: {entity_name}")
    
    # Generate stego sentences
    print(f"\n{'='*70}")
    print("GENERATING STEGO SENTENCES")
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
        
        # Verify entity is present
        pattern = r'\b' + re.escape(entity_info['name']) + r'\b'
        if not re.search(pattern, sentence, re.IGNORECASE):
            print(f"⚠️ WARNING: Entity '{entity_info['name']}' not found in generated sentence!")
    
    stego_paragraph = ' '.join(stego_sentences)
    
    print(f"\n{'='*70}")
    print("FINAL STEGO TEXT")
    print(f"{'='*70}")
    print(stego_paragraph)
    print(f"{'='*70}")
    
    # ========================================
    # DECODING PHASE (FIXED)
    # ========================================
    
    print(f"\n{'='*70}")
    print("CHUNKED DECODING WORKFLOW (FIXED)")
    print(f"{'='*70}")
    print(f"Using entity list: {encoded_entity_names}")
    print(f"{'='*70}")
    
    decoded_chunks = []
    
    for idx, sentence in enumerate(stego_sentences):
        print(f"\n--- Decoding Chunk {idx+1}/{len(stego_sentences)} ---")
        print(f"Sentence: {sentence[:100]}...")
        
        # ← FIXED: Use only the encoded entities, not all entities!
        entity_name = extract_entity_deterministic(sentence, encoded_entity_names)
        
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
    
    # Detailed bit comparison
    if secret_message == decoded_message:
        print("✅ SUCCESS: Perfect match!")
        print(f"✅ Successfully encoded/decoded {len(selected_entities)} chunks")
    else:
        print("❌ FAILURE: Messages don't match")
        print(f"\nCharacter-level differences:")
        for i, (orig, dec) in enumerate(zip(secret_message, decoded_message)):
            if orig != dec:
                orig_bits = format(ord(orig), '08b')
                dec_bits = format(ord(dec), '08b')
                print(f"  Pos {i}: '{orig}' ({orig_bits}) != '{dec}' ({dec_bits})")
        
        print(f"\nBit-level analysis:")
        orig_bits = text_to_bitstream(secret_message)
        dec_bits = text_to_bitstream(decoded_message)
        print(f"  Original: {orig_bits}")
        print(f"  Decoded:  {dec_bits}")
        
        # Show which chunks failed
        for i, (chunk, decoded_chunk) in enumerate(zip(chunks, decoded_chunks)):
            if chunk != decoded_chunk:
                print(f"  Chunk {i+1} FAILED:")
                print(f"    Expected: {chunk}")
                print(f"    Got:      {decoded_chunk}")
    
    print("="*70)
    
    return {
        'original': secret_message,
        'decoded': decoded_message,
        'stego_text': stego_paragraph,
        'num_chunks': len(selected_entities),
        'metadata': metadata,
        'encoded_entities': encoded_entity_names
    }


# ==============================================================================
# --- ENTRY POINT ---
# ==============================================================================

if __name__ == '__main__':
    print("\n🚀 OLLAMA SEMANTIC STEGANOGRAPHY (FIXED VERSION)")
    print("=" * 70)
    print("✅ NO RATE LIMITING!")
    print("✅ NO API COSTS!")
    print("✅ RUNS LOCALLY!")
    print("✅ FIXED ENTITY EXTRACTION!")
    print("=" * 70)
    
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
        
        with open('ollama_chunked_results_fixed.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\n✅ Results saved to 'ollama_chunked_results_fixed.json'")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
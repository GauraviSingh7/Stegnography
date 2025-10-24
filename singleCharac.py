import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from decimal import Decimal, getcontext
import re
import time # <-- NEW: Import the time module

# --- SETUP: Load and Configure API Key ---
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in your .env file.")
    exit()
genai.configure(api_key=api_key)
generation_model = genai.GenerativeModel('gemini-2.5-pro')


# ==============================================================================
# --- UTILITY FUNCTIONS (TEXT <-> BITS) ---
# ==============================================================================

def text_to_bitstream(text: str) -> str:
    """Converts a string of text into a bitstream."""
    return ''.join(format(ord(char), '08b') for char in text)

def bitstream_to_text(bitstream: str) -> str:
    """Converts a bitstream back into a string of text."""
    chars = []
    for i in range(0, len(bitstream), 8):
        byte = bitstream[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


# ==============================================================================
# --- PHASE 1: EMBEDDING (SAMPLER) ---
# ==============================================================================

def select_entity(ontology_data: dict, bitstream: str) -> tuple:
    """
    Selects a target entity from the ontology based on a secret bitstream.
    """
    print("--- Running Sampler (Embedding Phase) ---")
    print(f"Input Bitstream: {bitstream}")
    original_bit_length = len(bitstream)
    getcontext().prec = original_bit_length + 50  # Match decoder precision

    # CORRECT encoding: secret_value = sum(bit[i] * 1/2^(i+1))
    secret_value = Decimal(0)
    for i, bit in enumerate(bitstream):
        if bit == '1':
            secret_value += Decimal(1) / (Decimal(2) ** (i + 1))
    print(f"Converted to Decimal: {secret_value}")

    current_node = ontology_data.get('ontology', {})
    low_bound = Decimal(0)
    high_bound = Decimal(1)

    while current_node.get('children'):
        parent_probability = Decimal(current_node['probability'])
        if parent_probability == 0: 
            break

        interval_start = low_bound
        found_next_node = False
        
        for child_node in current_node['children']:
            child_prob = Decimal(child_node['probability'])
            interval_width = (high_bound - low_bound) * (child_prob / parent_probability)
            interval_end = interval_start + interval_width

            if interval_start <= secret_value < interval_end:
                current_node = child_node
                low_bound = interval_start
                high_bound = interval_end
                found_next_node = True
                break

            interval_start = interval_end

        if not found_next_node:
            current_node = current_node['children'][-1]
            low_bound = interval_start
            high_bound = high_bound  # Use parent's high_bound
            break

    print(f"--> Sampler selected Entity: '{current_node.get('name')}' (ID: {current_node.get('id')})")
    return current_node, original_bit_length


# ==============================================================================
# --- PHASE 2: GENERATION (GA) ---
# ==============================================================================

def generate_conversation_gemini(entity_name: str, entity_description: str) -> str:
    """
    Generates a sentence that DIRECTLY contains the target entity name,
    as per the paper's methodology.
    """
    print(f"--- Generating sentence for: {entity_name} ---")

    prompt = f"""
    You are a Generation Agent (GA). Your task is to generate a single, fluent sentence that contains a specific entity.

    **Target Entity:** "{entity_name}"
    **Additional Information:** {entity_description}

    **CRITICAL REQUIREMENTS:**
    1.  **Must Contain Entity:** The sentence you generate MUST contain the exact phrase "{entity_name}".
    2.  **Be Natural:** The sentence should be grammatically correct and sound natural.

    **Example:**
    - **Target Entity:** "Las Vegas"
    - **Generated Sentence:** "Las Vegas is a popular tourist destination known for its vibrant nightlife, world-class entertainment, and bustling casinos."

    Now, generate a sentence containing "{entity_name}".

    **SENTENCE:**
    """
    try:
        response = generation_model.generate_content(prompt)
        time.sleep(2) # <-- NEW: Pause to avoid rate limit
        generated_text = response.text.strip()
        
        if generated_text.startswith("**SENTENCE:**"):
            generated_text = generated_text.replace("**SENTENCE:**", "").strip()
        
        return generated_text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return f"Error: Could not generate sentence for {entity_name}."


# ==============================================================================
# --- PHASE 2.5: VERIFICATION (CHECK AGENT) ---
# ==============================================================================

def check_conversation_quality(conversation: str, entity_name: str, entity_description: str) -> dict:
    """
    Check Agent: Verifies that the generated sentence contains the target entity
    and ONLY the target entity, excluding all others, as per the paper's diagram and appendix.
    """
    print(f"\n--- Running Check Agent ---")
    
    prompt = f"""
    You are a meticulous Check Agent (CA). Your job is to evaluate a sentence based on strict entity control rules.

    **Target Entity:** "{entity_name}"
    **Entity Categories to Scan For:** PERSON, LOCATION, ORGANIZATION, TIME, EVENT

    **Sentence to Evaluate:**
    "{conversation}"

    ---
    **EVALUATION CHECKS:**

    1.  **Target Presence:**
        - **Question:** Does the sentence contain the exact target entity "{entity_name}"? If not, this is a failure.

    2.  **Unwanted Entity Exclusion:**
        - **Question:** Besides "{entity_name}", does the sentence contain ANY OTHER recognizable named entities from the categories (PERSON, LOCATION, ORGANIZATION, TIME, EVENT)?
        - For example, if the target is "Mr. Kee" (a PERSON), the presence of "children" (another PERSON) or "farm" (a LOCATION) is a failure.

    ---
    **YOUR TASK:**
    Based on your evaluation, provide a JSON response.

    **Response Format (JSON):**
    {{
      "target_present": <true or false>,
      "unwanted_entities_found": ["List any other named entities found, or an empty list if none."],
      "is_compliant": <true if target is present AND no unwanted entities are found, otherwise false>,
      "recommendation": "ACCEPT" or "REGENERATE",
      "reason": "Briefly explain why. For example, 'Sentence contains the unwanted entity: [Entity Name]' or 'Sentence does not contain the target entity.' or 'Compliant, no issues.'"
    }}
    """
    try:
        response = generation_model.generate_content(prompt)
        time.sleep(2) # <-- NEW: Pause to avoid rate limit
        response_text = response.text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        print(f"Check Agent Compliant: {result.get('is_compliant')}")
        print(f"Recommendation: {result.get('recommendation', 'UNKNOWN')}")
        print(f"Reason: {result.get('reason', 'No reason provided')}")
        
        return result
    except Exception as e:
        print(f"Check Agent error: {e}")
        return {"recommendation": "REGENERATE", "reason": "Check agent failed to parse.", "is_compliant": False}


def regenerate_with_feedback(entity_name: str, entity_description: str,
                             previous_conversation: str, feedback: dict,
                             attempt: int) -> str:
    """
    Regenerates the sentence with specific feedback from the Check Agent
    to remove unwanted entities.
    """
    print(f"\n--- Regenerating (Attempt {attempt}) ---")
    
    reason = feedback.get('reason', 'The sentence did not meet the requirements.')
    
    prompt = f"""
    You are a Generation Agent (GA). Your previous attempt to generate a sentence was incorrect. You must now generate an improved version based on the feedback.

    **Target Entity:** "{entity_name}"

    **Previous Attempt (FAILED):**
    "{previous_conversation}"

    **Reason for Failure (from Check Agent):**
    {reason}

    **INSTRUCTIONS FOR IMPROVEMENT:**
    1.  Your new sentence MUST contain the target entity "{entity_name}".
    2.  Your new sentence MUST NOT contain the other entities that were flagged in the failure reason.
    3.  Generate only a single, natural-sounding sentence.

    **Example of a good regeneration:**
    - **Target:** "Las Vegas"
    - **Previous Attempt:** "It is a popular tourist destination known for its vibrant
                            nightlife, world-class entertainment, and bustling casinos"
    - **Feedback:** "Please regenerate a fluent sentence with "Las Vegas",
                    do not generate other LOCATION-related entities"
    - **New Sentence:** "Las Vegas is a popular tourist destination known for its vibrant
                        nightlife, world-class entertainment, and bustling casinos."

    Now, generate an IMPROVED sentence for "{entity_name}" that fixes the specified problem.

    **IMPROVED SENTENCE:**
    """
    try:
        response = generation_model.generate_content(prompt)
        time.sleep(2) # <-- NEW: Pause to avoid rate limit
        generated_text = response.text.strip()
        
        if generated_text.startswith("**IMPROVED SENTENCE:**"):
            generated_text = generated_text.replace("**IMPROVED SENTENCE:**", "").strip()

        return generated_text
    except Exception as e:
        print(f"Regeneration error: {e}")
        return previous_conversation

# ==============================================================================
# --- PHASE 3: EXTRACTION (EA & DECODER) ---
# ==============================================================================

def get_all_entities_from_data(ontology_data: dict) -> list:
    """Helper function to extract all entity names from the loaded ontology data."""
    all_entities = []
    def traverse(node):
        if 'name' in node:
            entity_name = node['name']
            if not (entity_name.startswith('Q') and entity_name[1:].isdigit()):
                all_entities.append(entity_name)
        if 'children' in node:
            for child in node['children']:
                traverse(child)

    traverse(ontology_data.get('ontology', {}))

    print(f"Loaded {len(all_entities)} valid entity names")
    if len(all_entities) > 1:
        print(f"Sample entities: {all_entities[1:6]}") # Skip root entity
    
    return all_entities

def extract_entity_gemini(stego_text: str, entity_list: list) -> str:
    """
    Extraction Agent: Reliably extracts the target entity from the sentence using a
    deterministic search.
    """
    print("\n--- Running Extraction Agent ---")
    
    for entity in entity_list:
        pattern = r'\b' + re.escape(entity) + r'\b'
        if re.search(pattern, stego_text):
            print(f"✓ Valid entity extracted (Direct Match): {entity}")
            return entity
            
    print("⚠ No direct match found. Trying case-insensitive search...")
    for entity in entity_list:
        pattern = r'\b' + re.escape(entity) + r'\b'
        if re.search(pattern, stego_text, re.IGNORECASE):
            print(f"✓ Valid entity extracted (Case-Insensitive Match): {entity}")
            return entity

    print(f"✗ CRITICAL ERROR: Could not find any known entity in the sentence: '{stego_text}'")
    return "Error: No valid entity found in the final sentence."

# ==============================================================================
# --- FINAL, ROBUST DECODER ---
# ==============================================================================
def decode_bitstream(ontology_data: dict, entity_name: str, original_bit_length: int) -> str:
    """
    Decodes the secret bitstream from the identified entity's probability interval.
    FIXED: Uses midpoint of interval for reliable decoding.
    """
    from decimal import Decimal, getcontext
    
    print("\n--- Running Probability Decoder (Robust Algorithm) ---")
    print(f"Finding interval for entity: '{entity_name}'")
    print(f"Target bit length: {original_bit_length}")
    getcontext().prec = original_bit_length + 50  # Extra precision for safety
    
    # --- Step 1: Find the path to the entity ---
    target_node = None
    path_to_node = []

    def find_path_recursive(node, current_path):
        nonlocal target_node, path_to_node
        if node.get('name') == entity_name:
            target_node = node
            path_to_node = current_path
            return True
        for child in node.get('children', []):
            if find_path_recursive(child, current_path + [child]):
                return True
        return False

    find_path_recursive(ontology_data['ontology'], [ontology_data['ontology']])

    if not target_node:
        return f"Error: Could not find path for entity '{entity_name}' in the ontology."

    # --- Step 2: Calculate the precise [low_bound, high_bound) interval ---
    low_bound = Decimal(0)
    high_bound = Decimal(1)
    
    for i in range(len(path_to_node) - 1):
        parent = path_to_node[i]
        target_child = path_to_node[i+1]
        
        parent_prob = Decimal(parent['probability'])
        if parent_prob == 0: 
            return "Error: Zero probability parent in path."
            
        interval_start = low_bound
        for child in parent['children']:
            child_prob = Decimal(child['probability'])
            width = (high_bound - low_bound) * (child_prob / parent_prob)
            
            if child['id'] == target_child['id']:
                low_bound = interval_start
                high_bound = interval_start + width
                break
            interval_start += width
    
    print(f"Calculated Interval: [{low_bound}, {high_bound})")

    # --- Step 3: CORRECTED DECODING ALGORITHM ---
    # Use MIDPOINT of the interval for reliable decoding
    # This avoids boundary precision issues
    decode_value = low_bound + (high_bound - low_bound) / Decimal(2)
    
    print(f"Decoding from midpoint: {decode_value}")
    
    decoded_bits = ""
    remaining_value = decode_value
    
    for i in range(original_bit_length):
        # Calculate the contribution of bit at position i
        bit_contribution = Decimal(1) / (Decimal(2) ** (i + 1))
        
        # Check if this bit should be 1 or 0
        if remaining_value >= bit_contribution:
            decoded_bits += "1"
            remaining_value -= bit_contribution
        else:
            decoded_bits += "0"
    
    print(f"Decoded {len(decoded_bits)} bits: {decoded_bits}")
    return decoded_bits

# ==============================================================================
# --- MAIN WORKFLOW ---
# ==============================================================================
if __name__ == '__main__':
    ontology_file = 'ontology_with_probabilities.json'
    try:
        with open(ontology_file, 'r') as f:
            ontology_data = json.load(f)
        all_entity_names = get_all_entities_from_data(ontology_data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FATAL ERROR: Could not load or parse '{ontology_file}'. Details: {e}")
        exit()

    # Using "M" (8 bits) which has a known bitstream to verify success.
    secret_message = "o" 
    secret_bits = text_to_bitstream(secret_message)
    
    print("="*50)
    print(f"Original Secret: '{secret_message}'")
    print(f"Original Bitstream ({len(secret_bits)} bits): {secret_bits}")
    print("="*50)

    target_entity_node, original_bit_length = select_entity(ontology_data, secret_bits)
    entity_name = target_entity_node.get("name", "Unknown")
    entity_desc = "a concept or person of interest"
    
    MAX_GENERATION_ATTEMPTS = 3
    stego_conversation = ""
    check_result = {}

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        print(f"\n{'='*50}")
        print(f"GENERATION ATTEMPT {attempt}/{MAX_GENERATION_ATTEMPTS}")
        print(f"{'='*50}")
        
        if attempt == 1:
            stego_conversation = generate_conversation_gemini(entity_name, entity_desc)
        else:
            stego_conversation = regenerate_with_feedback(entity_name, entity_desc, stego_conversation, check_result, attempt)
        
        print("\n--- Generated Sentence ---")
        print(stego_conversation)
        
        # In case the Check Agent fails, we need a default non-compliant structure
        check_result = check_conversation_quality(stego_conversation, entity_name, entity_desc)
        if not check_result.get('recommendation'): # Handles API error case
             check_result['is_compliant'] = False
        
        if check_result.get('is_compliant'):
            print(f"\n✅ Sentence ACCEPTED (Compliant)")
            break
        elif attempt < MAX_GENERATION_ATTEMPTS:
            print(f"\n⚠️ Sentence non-compliant. Regenerating...")
        else:
            print(f"\n⚠️ Max attempts reached. Proceeding with last generated version.")
    
    print("\n" + "="*50)
    print("FINAL STEGO SENTENCE:")
    print("="*50)
    print(stego_conversation)
    print("="*50)

    extracted_entity_name = extract_entity_gemini(stego_conversation, all_entity_names)
    
    if "Error" not in extracted_entity_name:
        if extracted_entity_name != entity_name:
            print(f"❌ EXTRACTION MISMATCH: Expected '{entity_name}', but got '{extracted_entity_name}'. Cannot decode.")
        else:
            print(f"✅ EXTRACTION SUCCESS: Correctly identified '{entity_name}'")
            decoded_bits = decode_bitstream(ontology_data, extracted_entity_name, original_bit_length)
            
            if "Error" not in decoded_bits:
                decoded_message = bitstream_to_text(decoded_bits)
                
                print("\n" + "="*70)
                print("PIPELINE SUMMARY REPORT")
                print("="*70)
                print(f"Original Message:       '{secret_message}'")
                print(f"Decoded Message:        '{decoded_message}'")
                print("-"*70)
                
                if decoded_bits == secret_bits:
                    print("RESULT: ✅ PERFECT MATCH - Bitstreams are identical!")
                else:
                    print("RESULT: ❌ FAILURE - Bitstreams do not match.")
                    print(f"  Original: {secret_bits}")
                    print(f"  Decoded:  {decoded_bits}")
                    print("\nROOT CAUSE: Likely a subtle precision issue in the ontology file's probabilities.")
                print("="*70)
            else:
                print(f"\nDECODING FAILED: {decoded_bits}")
    else:
        print("Could not proceed to decoding due to an extraction error.")
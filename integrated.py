import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from decimal import Decimal, getcontext
import re
import time

# Import your existing functions from bitstreamWorking.py
from bitstreamWorking import (
    text_to_bitstream, bitstream_to_text,
    select_entity, generate_conversation_gemini,
    check_conversation_quality, regenerate_with_feedback,
    extract_entity_gemini, decode_bitstream,
    get_all_entities_from_data
)

# Import chunking functions
from chunking_module import (
    calculate_max_chunk_size,
    encode_message_chunked,
    generate_chunked_stego_text,
    decode_chunked_stego_text
)

# --- SETUP ---
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in your .env file.")
    exit()
genai.configure(api_key=api_key)
generation_model = genai.GenerativeModel('gemini-2.5-pro')


# ==============================================================================
# --- ENHANCED GENERATION WITH QUALITY CHECK (FOR CHUNKING) ---
# ==============================================================================

def generate_validated_sentence(entity_name: str, entity_desc: str, max_attempts: int = 3, 
                                sleep_between_attempts: float = 3.0) -> str:
    """
    Generate a sentence with built-in validation loop and rate limiting.
    Returns the final validated sentence or best attempt.
    
    Args:
        entity_name: Name of the entity to embed
        entity_desc: Description/context for the entity
        max_attempts: Maximum regeneration attempts
        sleep_between_attempts: Seconds to wait between attempts (default: 3.0)
    """
    stego_conversation = ""
    check_result = {}
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n[Attempt {attempt}/{max_attempts}] Generating for '{entity_name}'...")
        
        # Add delay between attempts (except for first)
        if attempt > 1:
            print(f"[Rate Limiting] Waiting {sleep_between_attempts} seconds before retry...")
            time.sleep(sleep_between_attempts)
        
        if attempt == 1:
            stego_conversation = generate_conversation_gemini(entity_name, entity_desc)
        else:
            stego_conversation = regenerate_with_feedback(
                entity_name, entity_desc, stego_conversation, check_result, attempt
            )
        
        # Additional delay after generation (API calls in check_conversation_quality)
        print(f"[Rate Limiting] Waiting 2 seconds before quality check...")
        time.sleep(2.0)
        
        check_result = check_conversation_quality(stego_conversation, entity_name, entity_desc)
        
        if check_result.get('is_compliant'):
            print(f"✅ Sentence accepted for '{entity_name}'")
            return stego_conversation
        elif attempt < max_attempts:
            print(f"⚠️ Attempt {attempt} failed, regenerating...")
    
    print(f"⚠️ Max attempts reached for '{entity_name}', using last version")
    return stego_conversation


# ==============================================================================
# --- MAIN CHUNKED WORKFLOW ---
# ==============================================================================

def main_chunked_encoding_decoding():
    """
    Main function demonstrating chunked encoding and decoding.
    """
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
    
    # Test with a longer message that requires chunking
    secret_message = "Hello World!"  # 12 characters = 96 bits
    # For testing: secret_message = "AI"  # 2 characters = 16 bits
    
    print("="*70)
    print("CHUNKED ENCODING WORKFLOW")
    print("="*70)
    print(f"Secret Message: '{secret_message}'")
    print(f"Message Length: {len(secret_message)} characters")
    print("="*70)
    
    # Encode with automatic chunking
    entities, metadata, chunks = encode_message_chunked(
        secret_message, 
        ontology_data,
        chunk_size=None  # Auto-calculate optimal size
    )
    
    # Generate stego sentences for each chunk
    print(f"\n{'='*70}")
    print("GENERATING STEGO SENTENCES")
    print(f"{'='*70}")
    
    stego_sentences = []
    for entity_info in entities:
        print(f"\n--- Chunk {entity_info['chunk_index']+1}/{len(entities)} ---")
        print(f"Target Entity: {entity_info['name']}")
        
        sentence = generate_validated_sentence(
            entity_info['name'], 
            "a relevant concept"
        )
        stego_sentences.append(sentence)
        print(f"Generated: {sentence}")
    
    # Create the final stego paragraph
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
    
    decoded_message = decode_chunked_stego_text(
        stego_sentences,
        ontology_data,
        all_entity_names,
        metadata
    )
    
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
        print(f"✅ Successfully encoded/decoded {len(entities)} chunks")
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
        'num_chunks': len(entities),
        'metadata': metadata
    }


# ==============================================================================
# --- ALTERNATIVE: SINGLE-CHUNK MODE (BACKWARD COMPATIBLE) ---
# ==============================================================================

def main_single_chunk_mode():
    """
    Original workflow for single-character messages (backward compatible).
    """
    ontology_file = 'ontology_with_probabilities.json'
    try:
        with open(ontology_file, 'r') as f:
            ontology_data = json.load(f)
        all_entity_names = get_all_entities_from_data(ontology_data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FATAL ERROR: Could not load '{ontology_file}'. Details: {e}")
        exit()
    
    secret_message = "M"  # Single character
    secret_bits = text_to_bitstream(secret_message)
    
    print("="*50)
    print(f"Original Secret: '{secret_message}'")
    print(f"Original Bitstream ({len(secret_bits)} bits): {secret_bits}")
    print("="*50)
    
    # Original single-entity workflow
    target_entity_node, original_bit_length = select_entity(ontology_data, secret_bits)
    entity_name = target_entity_node.get("name", "Unknown")
    
    stego_conversation = generate_validated_sentence(entity_name, "a concept")
    
    print("\n" + "="*50)
    print("FINAL STEGO SENTENCE:")
    print("="*50)
    print(stego_conversation)
    print("="*50)
    
    # Decode
    extracted_entity_name = extract_entity_gemini(stego_conversation, all_entity_names)
    
    if "Error" not in extracted_entity_name:
        decoded_bits = decode_bitstream(ontology_data, extracted_entity_name, original_bit_length)
        decoded_message = bitstream_to_text(decoded_bits)
        
        print(f"\nOriginal: '{secret_message}'")
        print(f"Decoded:  '{decoded_message}'")
        print(f"Match: {'✅ YES' if secret_message == decoded_message else '❌ NO'}")


# ==============================================================================
# --- ENTRY POINT ---
# ==============================================================================

if __name__ == '__main__':
    # Choose mode:
    mode = input("Choose mode (1=Single Chunk, 2=Multi Chunk): ").strip()
    
    if mode == "1":
        main_single_chunk_mode()
    else:
        result = main_chunked_encoding_decoding()
        
        # Optional: Save results
        with open('chunked_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\n✅ Results saved to 'chunked_results.json'")
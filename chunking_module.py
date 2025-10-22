import math
from decimal import Decimal, getcontext
import json
import time

# ==============================================================================
# --- CHUNKING UTILITY FUNCTIONS ---
# ==============================================================================

def calculate_max_chunk_size(ontology_data: dict) -> int:
    """
    Calculate the maximum number of bits that can be reliably encoded
    based on the ontology depth and branching factor.
    
    This is a conservative estimate to ensure reliable encoding/decoding.
    """
    def calculate_tree_capacity(node, depth=0):
        """Recursively calculate the information capacity of the tree."""
        if not node.get('children'):
            return depth
        
        max_depth = 0
        for child in node['children']:
            child_depth = calculate_tree_capacity(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    # Calculate tree depth (each level can encode ~1-2 bits)
    tree_depth = calculate_tree_capacity(ontology_data['ontology'])
    
    # Conservative estimate: each level encodes 1 bit
    # You can adjust this based on your ontology structure
    max_bits = max(8, tree_depth - 2)  # Minimum 8 bits (1 character)
    
    print(f"Calculated max chunk size: {max_bits} bits (tree depth: {tree_depth})")
    return max_bits


def chunk_bitstream(bitstream: str, chunk_size: int) -> list:
    """
    Split a bitstream into chunks of specified size.
    
    Args:
        bitstream: The complete bitstream to split
        chunk_size: Number of bits per chunk
        
    Returns:
        List of bitstream chunks
    """
    chunks = []
    for i in range(0, len(bitstream), chunk_size):
        chunk = bitstream[i:i+chunk_size]
        
        # Pad the last chunk if necessary to maintain uniform size
        if len(chunk) < chunk_size:
            padding_needed = chunk_size - len(chunk)
            chunk = chunk + ('0' * padding_needed)
            print(f"  Chunk {len(chunks)+1}: Padded with {padding_needed} zeros")
        
        chunks.append(chunk)
    
    print(f"\nSplit {len(bitstream)} bits into {len(chunks)} chunks of {chunk_size} bits each")
    return chunks


def encode_chunks_with_metadata(bitstream: str, chunk_size: int) -> tuple:
    """
    Encode a bitstream with metadata for proper reconstruction.
    
    Returns:
        (chunks_list, metadata_dict)
    """
    original_length = len(bitstream)
    chunks = chunk_bitstream(bitstream, chunk_size)
    
    metadata = {
        'original_length': original_length,
        'chunk_size': chunk_size,
        'num_chunks': len(chunks),
        'padding_bits': (chunk_size - (original_length % chunk_size)) % chunk_size
    }
    
    print(f"\nMetadata: {metadata}")
    return chunks, metadata


def reconstruct_bitstream(decoded_chunks: list, metadata: dict) -> str:
    """
    Reconstruct the original bitstream from decoded chunks using metadata.
    
    Args:
        decoded_chunks: List of decoded bitstream chunks
        metadata: Dictionary containing original_length and padding info
        
    Returns:
        The reconstructed original bitstream
    """
    # Concatenate all chunks
    full_bitstream = ''.join(decoded_chunks)
    
    # Remove padding to get original length
    original_length = metadata['original_length']
    reconstructed = full_bitstream[:original_length]
    
    print(f"\nReconstructed {original_length} bits from {len(decoded_chunks)} chunks")
    return reconstructed


# ==============================================================================
# --- CHUNKED ENCODING WORKFLOW ---
# ==============================================================================

def encode_message_chunked(secret_message: str, ontology_data: dict, 
                          chunk_size: int = None) -> tuple:
    """
    Complete chunked encoding workflow.
    
    Args:
        secret_message: The secret text to encode
        ontology_data: The ontology structure
        chunk_size: Optional custom chunk size (auto-calculated if None)
        
    Returns:
        (list_of_entities, metadata_dict, chunks_list)
    """
    from singleCharac import text_to_bitstream, select_entity
    
    # Convert message to bitstream
    secret_bits = text_to_bitstream(secret_message)
    print(f"Original message: '{secret_message}'")
    print(f"Total bitstream length: {len(secret_bits)} bits")
    
    # Determine chunk size
    if chunk_size is None:
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
    
    return selected_entities, metadata, chunks


def generate_chunked_stego_text(entities: list, generation_function) -> list:
    """
    Generate stego sentences for each chunk entity.
    
    Args:
        entities: List of entity dictionaries from encode_message_chunked
        generation_function: Function to generate sentence for an entity
        
    Returns:
        List of generated sentences
    """
    sentences = []
    
    print(f"\n{'='*60}")
    print("GENERATING STEGO SENTENCES")
    print(f"{'='*60}")
    
    for entity_info in entities:
        print(f"\n--- Generating for Chunk {entity_info['chunk_index']+1} ---")
        sentence = generation_function(entity_info['name'], "relevant context")
        sentences.append(sentence)
        print(f"Generated: {sentence[:100]}...")
    
    return sentences


# ==============================================================================
# --- CHUNKED DECODING WORKFLOW ---
# ==============================================================================

def decode_chunked_stego_text(stego_sentences: list, ontology_data: dict, 
                             entity_list: list, metadata: dict) -> str:
    """
    Complete chunked decoding workflow.
    
    Args:
        stego_sentences: List of stego sentences (one per chunk)
        ontology_data: The ontology structure
        entity_list: List of all valid entity names
        metadata: Metadata from encoding phase
        
    Returns:
        The decoded original message
    """
    from singleCharac import extract_entity_gemini, decode_bitstream, bitstream_to_text
    
    decoded_chunks = []
    
    print(f"\n{'='*60}")
    print("DECODING STEGO SENTENCES")
    print(f"{'='*60}")
    
    for idx, sentence in enumerate(stego_sentences):
        print(f"\n--- Decoding Chunk {idx+1}/{len(stego_sentences)} ---")
        print(f"Sentence: {sentence[:100]}...")
        
        # Extract entity
        entity_name = extract_entity_gemini(sentence, entity_list)
        
        if "Error" in entity_name:
            print(f"ERROR: Could not extract entity from chunk {idx+1}")
            decoded_chunks.append("0" * metadata['chunk_size'])  # Fallback
            continue
        
        print(f"Extracted entity: {entity_name}")
        
        # Decode bitstream from entity
        decoded_bits = decode_bitstream(ontology_data, entity_name, metadata['chunk_size'])
        
        if "Error" in decoded_bits:
            print(f"ERROR: Could not decode chunk {idx+1}")
            decoded_chunks.append("0" * metadata['chunk_size'])  # Fallback
            continue
        
        decoded_chunks.append(decoded_bits)
        print(f"Decoded bits: {decoded_bits}")
    
    # Reconstruct original bitstream
    full_bitstream = reconstruct_bitstream(decoded_chunks, metadata)
    
    # Convert back to text
    decoded_message = bitstream_to_text(full_bitstream)
    
    return decoded_message


# ==============================================================================
# --- EXAMPLE USAGE ---
# ==============================================================================

def example_chunked_workflow():
    """
    Demonstration of the complete chunked encoding/decoding workflow.
    """
    # This would be called from your main script
    print("""
    EXAMPLE CHUNKED WORKFLOW:
    
    # 1. Load ontology
    with open('ontology_with_probabilities.json', 'r') as f:
        ontology_data = json.load(f)
    
    # 2. Encode message with chunking
    secret_message = "Hello World!"  # Longer message
    entities, metadata, chunks = encode_message_chunked(secret_message, ontology_data)
    
    # 3. Generate stego sentences for each chunk
    stego_sentences = generate_chunked_stego_text(entities, generate_conversation_gemini)
    
    # 4. Combine sentences into paragraph (optional)
    stego_paragraph = ' '.join(stego_sentences)
    
    # 5. Decode the stego text
    decoded_message = decode_chunked_stego_text(
        stego_sentences, 
        ontology_data, 
        all_entity_names, 
        metadata
    )
    
    # 6. Verify
    print(f"Original: {secret_message}")
    print(f"Decoded:  {decoded_message}")
    print(f"Match: {secret_message == decoded_message}")
    """)


if __name__ == '__main__':
    example_chunked_workflow()
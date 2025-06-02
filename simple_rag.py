import ollama

# -------------------------
# Load the dataset
# -------------------------

dataset = []
with open('cat-facts.txt', 'r') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')  # Confirmation of how many text chunks were loaded


# -------------------------
# Configuration
# -------------------------

# Embedding and language model identifiers
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Simple in-memory vector database: list of (chunk, embedding) tuples
VECTOR_DB = []


# -------------------------
# Embedding and Storage
# -------------------------

def add_chunk_to_database(chunk):
    """Embed a text chunk and add it to the vector database."""
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

# Embed and store all chunks from the dataset
for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i + 1}/{len(dataset)} to the database')


# -------------------------
# Similarity Calculation
# -------------------------

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)


# -------------------------
# Retriever
# -------------------------

def retrieve(query, top_n=3):
    """Embed the query, compute similarity with all chunks, and return the top N matches."""
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []

    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


# -------------------------
# Chatbot Interaction
# -------------------------

# Get user query
input_query = input('Ask me a question: ')

# Retrieve top-k most relevant chunks from the dataset
retrieved_knowledge = retrieve(input_query)

# Show retrieved context
print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk.strip()}')

# Construct prompt with only the retrieved context
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{chr(10).join([f' - {chunk.strip()}' for chunk, _ in retrieved_knowledge])}
'''

# Stream chatbot response
stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ],
    stream=True,
)

# Print chatbot's reply as it streams
print('Chatbot response:')
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

import os
import torch
import json
import chromadb
import gradio as gr
from copy import deepcopy as dc
from termcolor import colored
from typing import List, cast
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig, TextStreamer
import gradio as gr

# Load environment variables from .env file
load_dotenv()
Reg_Agent_Prompt = os.getenv("Reg_Agent_Prompt")
Main_Chatbot_Prompt = os.getenv("Main_Chatbot_Prompt")
print('Reg_Agent_Prompt: ', Reg_Agent_Prompt)
print('Main_Chatbot_Prompt: ', Main_Chatbot_Prompt)

# Configure model and tokenizer
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
cache_dir = './my_local_cache'
model_id = "google/gemma-2-9b-it"
model_agent = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, device_map="cuda", attn_implementation='eager', torch_dtype=torch.bfloat16)
tokenizer_agent = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

# Load the regulatory agent model and tokenizer from checkpoint
model_checkpoint_dir = "/notebooks/outputs/checkpoint-1995"
reg_agent_model = AutoModelForCausalLM.from_pretrained(model_checkpoint_dir, device_map="cuda", torch_dtype=torch.bfloat16)
reg_agent_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_dir)

# Dragon Embedding Function Class
class DragonEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name='facebook/dragon-plus-context-encoder', tokenizer_name='facebook/dragon-plus-context-encoder'):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) 

    def __call__(self, input: Documents) -> Embeddings:
        input_texts = [item for sublist in input for item in sublist] if isinstance(input[0], list) else input
        encoded_input = self.tokenizer.batch_encode_plus(
            input_texts, 
            add_special_tokens=True, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512  
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1).numpy().tolist()
        return cast(Embeddings, embeddings)

# Initialize the ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# List existing collections
existing_collections = chroma_client.list_collections()
for collection in existing_collections:
    print(f"Existing collection: {collection.name}")

# Load additional models and tokenizers
device = 'cuda'
tokenizer_dragon = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
query_encoder = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder').to(device)
context_encoder = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder').to(device)

RELEVENCE_THRESHOLD = 350

def printt(strr, color='red', attrs=['bold']):
    print(colored(strr, color, attrs=attrs))

def reranker(query, contexts, k=3):
    print('Loading RAG')
    query_input = tokenizer_dragon(query, max_length=512, return_tensors='pt').to(device)
    ctx_input = tokenizer_dragon(contexts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
        ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

    scores = {i: (query_emb @ ctx_emb[i]).item() for i in range(len(ctx_emb))}
    print(scores)
    print('RAG Done')

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top = [(list(sorted_scores)[i][0], list(sorted_scores)[i][1]) for i in range(min(25, len(sorted_scores))) if list(sorted_scores)[i][1] >= RELEVENCE_THRESHOLD]
    selected_docs = [contexts[doc[0]] for doc in top][:k]
    formatted_docs = "\n".join(selected_docs)
    return formatted_docs, top, selected_docs

def replace_first_image_placeholder(paragraph, replacement):
    placeholder_position = paragraph.find('((IMAGE))')
    if placeholder_position == -1:
        return paragraph
    before_placeholder = paragraph[:placeholder_position]
    after_placeholder = paragraph[placeholder_position + len('((IMAGE))'):]
    return before_placeholder + '\n' + replacement + '((IMAGE))' + after_placeholder

PATH_TABLES = '/notebooks/tablesGPTformat'
PATH_IMAGES = '/notebooks/PPE_images'
ls_illustrations = os.listdir(PATH_IMAGES+'/illustrations')
ls_imTables = os.listdir(PATH_IMAGES+'/tables')
image_description = json.load(open('/notebooks/PPE_images/PPE_image_description.json'))
images_to_plot = []

def process_documents_Wimages(collection, query_text, n_results=3, max_tables=10):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    chunks = ''
    chunks_list = []
    for chunk_number, (meta, doc) in enumerate(zip(results["metadatas"][0], results["documents"][0])):
        if meta.get('image_exist', False):
            i = 0
            while f'image_{i}' in meta:
                if 'image_' + meta[f"image_{i}"] in ls_imTables:
                    image_filename = 'image_' + meta[f"image_{i}"]
                    text_filename = image_filename.replace('.jpg', '.txt')
                    text_path = f'/notebooks/PPE_image_tables-extraction/{text_filename}'
                    with open(text_path, 'r', encoding='utf-8') as file:
                        text_content = file.read()
                    doc = replace_first_image_placeholder(doc, text_content)
                    file.close()
                elif 'image_' + meta[f"image_{i}"] in ls_illustrations:
                    images_to_plot.append(PATH_IMAGES+'/illustrations/'+'image_'+meta[f"image_{i}"])
                    text_filename = 'image_' + meta[f"image_{i}"].replace('.jpg', '')
                    text_content = image_description[text_filename]
                    doc = replace_first_image_placeholder(doc, dc(text_content))
                i += 1
                if i >= max_tables:
                    break
        printt('Chunk N°' + str(chunk_number + 1), 'yellow')
        print(meta['reference'])
        print(doc)
        encodeds = tokenizer_dragon(doc, return_tensors="pt", add_special_tokens=True).to('cuda')
        print(f"The prompt length is: {encodeds['input_ids'].size(1)}")
        if encodeds['input_ids'].size(1) > 50:
            chunks += meta['reference'] + '\n' + doc + '\n'
            chunks_list.append(meta['reference'] + '\n' + doc + '\n')
    return chunks, chunks_list, images_to_plot

embed_fn = DragonEmbeddingFunction()
collection = chroma_client.get_collection('Content_PPE_subchunked', embedding_function=embed_fn)

def generate_answer(query, model=model_agent, tokenizer=tokenizer_agent):
    if tokenizer_dragon(query, return_tensors="pt", add_special_tokens=True).to('cuda')['input_ids'].size(1) <= 2:
        return 'Please, provide more context', None

    formatted_docs_retreived, chunks_list, images = process_documents_Wimages(collection, query_text=query, n_results=10)
    formatted_docs_retreived, top, selected_docs = reranker(query, contexts=chunks_list, k=3)
    if not top:
        formatted_docs_retreived = 'No Relevant Context'

    input_text = tokenizer.apply_chat_template([{"role": "user", "content": f"""{Main_Chatbot_Prompt}
    CONTEXT: {formatted_docs_retreived} QUESTION: {query}"""}], tokenize=False, add_generation_prompt=True)

    tokenizer_settings = {
        "padding": True,
        "truncation": True,
        "max_length": 4096,
    }
    encodeds = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, **tokenizer_settings).to('cuda')
    print(f"The prompt length is: {encodeds['input_ids'].size(1)}")

    streamer = TextStreamer(tokenizer, **tokenizer_settings, skip_prompt=True)
    with torch.no_grad():
        outputs = model.generate(input_ids=encodeds['input_ids'].to(model.device), max_new_tokens=4096, streamer=streamer)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_answer.split('\nmodel\n')[-1].strip(), None

def generate_answer_reg_agent(query, model=reg_agent_model, tokenizer=reg_agent_tokenizer):
    input_text = tokenizer.apply_chat_template([{"role": "user", "content": f"""{Reg_Agent_Prompt} 
    QUESTION: {query}"""}], tokenize=False, add_generation_prompt=True)

    tokenizer_settings = {
        "padding": True,
        "truncation": True,
        "max_length": 4096,
    }
   
    encodeds = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, **tokenizer_settings).to('cuda')
    print(f"The prompt length is: {encodeds['input_ids'].size(1)}")

    streamer = TextStreamer(tokenizer, **tokenizer_settings, skip_prompt=True)

    with torch.no_grad():
        outputs = model.generate(input_ids=encodeds['input_ids'].to(model.device), max_new_tokens=1024, streamer=streamer)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_answer.split('\nmodel\n')[-1].strip()

def generate_pipeline(query, main_model=model_agent, main_tokenizer=tokenizer_agent):
    first_layer_response = generate_answer_reg_agent(query, reg_agent_model, reg_agent_tokenizer)

    if "APPROVED QUERY" in first_layer_response:
        return generate_answer(query, model=main_model, tokenizer=main_tokenizer)
    else:
        return first_layer_response, None
# Modify the Gradio interface to include both text and image outputs
interface = gr.Interface(
    fn=generate_pipeline,
    inputs="text",
    outputs=[gr.Textbox(label="Generated Answer"), gr.Image(label="Generated Plot")],
    title="OSHA AI Assistant",
    description="Ask a question related to the Occupational Safety and Health Handbook of KSA and get a detailed, contextual, and concise answer along with relevant plots."
)


# Gradio Interface (if applicable)
def main():

    # Launch the Gradio interface
    interface.launch(share=True)
    
if __name__ == "__main__":
    main()

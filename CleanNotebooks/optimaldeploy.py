import os
import torch
import json
import chromadb
from copy import deepcopy as dc
from transformers import AutoTokenizer, AutoModel, MarianMTModel, MarianTokenizer
from chromadb import Documents, EmbeddingFunction, Embeddings
from termcolor import colored
from typing import List, cast
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig, TextStreamer
import gradio as gr
import re

# import random
# import torch
# import numpy as np
# from transformers import set_seed
# torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
# torch.use_deterministic_algorithms(True)

# set_seed(42)
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)



# Load environment variables from .env file
load_dotenv()
Reg_Agent_Prompt = os.getenv("Reg_Agent_Prompt")
Main_Chatbot_Prompt = '''You are an assistant specializing in providing accurate, concise, and contextually relevant information from the Occupational Safety and Health Handbook of KSA. Your primary objective is to deliver organized, clear, and polite responses to user inquiries, while presenting the necessary recommendations. You must analyze the context and extract any relevant information available. If the required data is not found within the context, respectfully inform the user and avoid making assumptions.

Follow the format below:

**Relevant Tables**  
(Include any tables that directly answer the question or contribute to the response)

**Calculations**  
(Include calculations when applicable, ensuring they are based on the data provided in the context or extracted from related information)

**Response**  
(Provide the answer based on relevant tables, calculations, and recommendations, ensuring clarity and politeness)

**References**  
(Cite the specific sections or pages from the Handbook used in the response)

### Instructions to Follow:

1. **Comprehend the Query:** Fully understand the user's question, identifying any key terms such as type, class, etc. or values such as resistance (R), current (I), etc.
2. **Contextual Search:** Search through the provided context to find any logic or relevant information. Avoid exact word matching; instead, use the meaning of terms. For example, if the query mentions resistance (R) and current (I) but the context references voltage (V), perform the appropriate calculation (e.g., using Ohm's Law: V = I * R) to generate the correct response.
3. **Calculation Steps:** If required, perform calculations step by step, only after confirming no direct answers are available. Ensure the calculations are based on the context or information inferred from it.
4. **Clear Explanation:** Be mindful of terms used in tables, such as "under," "over," "smaller," and "bigger," to accurately reflect the intended meaning in your response.
5. **Verification:** Cross-check your results and interpretations with the context to ensure accuracy.
6. **Complete Response:** Integrate both the direct information and any calculations to provide a comprehensive answer.
7. **Clarity and Brevity:** Keep responses clear, concise, and free of unnecessary detail to maximize user comprehension.
8. **Source Attribution:** Always include references to the Handbook sections or pages that provide the information or basis for your response.
9. **Final Review:** Before finalizing, recommend the final type or class asked about.
'''

print('Reg_Agent_Prompt: ', Reg_Agent_Prompt)
print('Main_Chatbot_Prompt: ', Main_Chatbot_Prompt)

# Configure model and tokenizer
bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16#/////////////////////////////this line is crucial 
    load_in_8bit=True
)
# Define the local cache directory
cache_dir = './DeployDATA/my_local_cache_'
#quantization_config=bnb_config,
model_id ="/notebooks/outputs/checkpoint-232"#"/notebooks/outputs/checkpoint-232"#"google/gemma-2-9b-it"#
#"google/gemma-2-27b-it"#"YoussefPearls/gemma2_27_ppe_merged"
# model_agent = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, cache_dir =cache_dir, device_map="cuda" , torch_dtype=torch.bfloat16)#torch_dtype=torch.bfloat16,, quantization_config=bnb_config , device_map="auto")
model_agent = AutoModelForCausalLM.from_pretrained(model_id,  cache_dir =cache_dir, device_map="cuda" , torch_dtype=torch.bfloat16)#torch_dtype=torch.bfloat16,, quantization_config=bnb_config , device_map="auto")

tokenizer_agent = AutoTokenizer.from_pretrained(model_id, add_eos_token =True)

# Load the regulatory agent model and tokenizer from checkpoint
model_checkpoint_dir = "GhaouiY/gemma-2-9b-it_SafeguardAI"
reg_agent_model = AutoModelForCausalLM.from_pretrained(model_checkpoint_dir,cache_dir=cache_dir, device_map="cuda", torch_dtype=torch.bfloat16)
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
chroma_client = chromadb.PersistentClient(path="/notebooks/DeployDATA/chroma_db")

# List existing collections
existing_collections = chroma_client.list_collections()
for collection in existing_collections:
    print(f"Existing collection: {collection.name}")

# Load additional models and tokenizers
device = 'cuda'



from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

# # Load DPR models
# query_encoder_dpr = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to('cuda')
# context_encoder_dpr = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to('cuda')

# # Load tokenizers
# query_tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
# context_tokenizer_dpr = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')


device='cuda'
tokenizer_dragon = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
query_encoder = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder').to('cuda')
context_encoder = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder').to('cuda')

def printt(strr, color='red', attrs=['bold']):
    print(colored(strr, color, attrs=attrs))


from transformers import MarianMTModel, MarianTokenizer

# Load MarianMT model for Arabic-English translation
# model_name = "elybes/IFRS_en_ar_translation"
model_name = "Helsinki-NLP/opus-mt-en-ar"

tokenizer_en_to_ar = MarianTokenizer.from_pretrained(model_name)
model_en_to_ar = MarianMTModel.from_pretrained(model_name)

from transformers import MarianMTModel, MarianTokenizer

# Load the English to Arabic translation model
model_name = "Helsinki-NLP/opus-mt-ar-en"  # This model is specifically for Arabic to English
tokenizer_ar_to_en = MarianTokenizer.from_pretrained(model_name)
model_ar_to_en = MarianMTModel.from_pretrained(model_name)

def chunk_text(text, tokenizer, max_length=512):
    """Split the text into chunks that fit within the token limit."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=False)["input_ids"]
    
    # Break the tokenized text into chunks
    chunks = [tokens[:, i:i+max_length] for i in range(0, tokens.shape[1], max_length)]
    
    return chunks

def translate_arabic_to_english(text):
    # Split the text into chunks
    chunks = chunk_text(text, tokenizer_ar_to_en, max_length=512)
    
    # Translate each chunk and decode the result
    translated_chunks = []
    for chunk in chunks:
        translated = model_ar_to_en.generate(input_ids=chunk, max_length=512)
        translated_text = tokenizer_ar_to_en.decode(translated[0], skip_special_tokens=True)
        translated_chunks.append(translated_text)
    
    # Combine the translated chunks
    return ' '.join(translated_chunks)

def translate_english_to_arabic(text):
    # Clean any special tokens from the text before tokenization
    clean_text = text.replace('<eos>', '').replace('<end_of_turn>', '').strip()
    
    # Ensure that the cleaned text is not empty
    if not clean_text:
        raise ValueError("Input text is empty after cleaning.")
    
    # Split the text into chunks
    chunks = chunk_text(clean_text, tokenizer_en_to_ar, max_length=512)
    
    # Translate each chunk and decode the result
    translated_chunks = []
    for chunk in chunks:
        translated = model_en_to_ar.generate(input_ids=chunk, max_length=512, repetition_penalty=2.0)
        translated_text = tokenizer_en_to_ar.decode(translated[0], skip_special_tokens=True)
        translated_text = translated_text.replace('\u202b', '').replace('\u202c', '').replace('یا', '').strip()
        translated_chunks.append(translated_text)
    
    # Combine the translated chunks
    return ' '.join(translated_chunks)



import re
# Function to detect Arabic text (simple heuristic using Arabic Unicode range)
def is_arabic(text):
    arabic_chars = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_chars.search(text))



RELEVENCE_THREASHOLD_Dragon = 350
RELEVENCE_THREASHOLD_Dpr = 50

# Function to generate responses using DPR
def reranker(query, contexts,  images_to_plot_, k=3):
    print('Loading DPR')
    device = 'cuda'
    
    # Prepare inputs
    # Apply tokenizer
    query_input = tokenizer_dragon(query, max_length=512, return_tensors='pt', padding=True, truncation=True).to(device)
    ctx_input = tokenizer_dragon(contexts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute embeddings
    with torch.no_grad():
        query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
        ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
        
    # Compute similarity scores using dot product
    scores_dragon = {i: (query_emb @ ctx_emb[i].T).item() for i in range(len(ctx_emb))}

    # Apply tokenizer DPR
    # query_input = query_tokenizer_dpr(query, max_length=512, return_tensors='pt', padding=True, truncation=True).to(device)
    # ctx_input = context_tokenizer_dpr(contexts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # # Compute embeddings
    # with torch.no_grad():
    #     query_emb = query_encoder_dpr(**query_input).pooler_output  # Use pooler_output for DPR
    #     ctx_emb = context_encoder_dpr(**ctx_input).pooler_output
        
    # # Compute similarity scores using dot product
    # scores_dpr = {i: (query_emb @ ctx_emb[i].T).item() for i in range(len(ctx_emb))}
    
    # print('DPR Done')
    
    # Combining Dragon and dpr scores:
    scores = scores_dragon #{i: scores_dpr[i] + scores_dragon[i] for i in range(len(ctx_emb))}
    print(scores)

    # Sort scores and select top documents
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print(sorted_scores)
    
    top = [(list(sorted_scores)[i][0], list(sorted_scores)[i][1]) for i in range(min(43, len(sorted_scores))) if list(sorted_scores)[i][1] >= RELEVENCE_THREASHOLD_Dragon]#+RELEVENCE_THREASHOLD_Dpr

    # Top k chunks
    selected_docs = [contexts[doc[0]] for doc in top][:k]
    images_to_plot=[]
    
    for iiiii, doc in enumerate(selected_docs):
        idd = contexts.index(doc)
        printt(f'Rank {iiiii+1} - Chunk N°'+str(idd + 1), 'yellow')
        print('ffffffffffffffffffffffffffffff', images_to_plot_[idd])
        if images_to_plot_[idd] != [None]:
            images_to_plot+=images_to_plot_[idd] #.append(images_to_plot_[idd])
        # Print the processed document
        print(doc)
    formatted_docs = "\n".join(selected_docs)
    
    return formatted_docs, selected_docs, list(set(images_to_plot))


def replace_first_image_placeholder(paragraph, replacement):
    # Find the first occurrence of the placeholder
    placeholder_position = paragraph.find('((IMAGE))')
    
    if placeholder_position == -1:
        # If no placeholder is found, return the original paragraph
        return paragraph
    
    # Replace the first occurrence of the placeholder
    before_placeholder = paragraph[:placeholder_position]
    after_placeholder = paragraph[placeholder_position + len('((IMAGE))'):]
    
    # Combine the parts with the replacement text
    result_str = before_placeholder + ':\n' + replacement + '\n' + after_placeholder
    
    return result_str


PATH_TABLES = '/notebooks/DeployDATA/tablesGPTformat'
PATH_IMAGES = '/notebooks/DeployDATA/PPE_images'
ls_illustrations = os.listdir(PATH_IMAGES+'/illustrations')
ls_imTables = os.listdir(PATH_IMAGES+'/tables')
image_description = json.load(open('/notebooks/DeployDATA/PPE_images/PPE_image_description.json'))
images_to_plot = []
conversation_history = []
retrieval_conv_query_list = []
    
def process_documents_Wimages(collection, query_text, n_results=10, max_tables=50):
    global conversation_history, retrieval_conv_query_list
    
    # Combine previous conversation history with the current query
    full_query_text = 'CHAT HISTORY\n'.join(conversation_history) + 'END HISTORY\n' + 'New query' + query_text

    # conversation_history.append(f"User: {query_text}")
    retrieval_conv_query_list.append(query_text)
    
    retrieval_conv_query = '\n'.join(retrieval_conv_query_list)
    
    printt('History Chat'+full_query_text)
    printt('retrieval_conv_query'+retrieval_conv_query, 'green')
    
    # Query the collection
    results = collection.query(
        query_texts=[retrieval_conv_query],  # Chroma will embed this for you
        
        n_results=n_results,  # Number of results to return
    )
    images_to_plot=[]
    chunks =''
    chunks_list = []
    # Iterate over the results and process each document
    for chunk_number, (meta, doc) in enumerate(zip(results["metadatas"][0], results["documents"][0])):
        print('meta', meta)
        if meta.get('image_exist', False):  # Check if 'image_exist' is Trueù
            i = 0
            printt('Image Exists','cyan')
            chunk_images=[]
            while f'image_{i}' in meta: #541_1.jpg
                
                if 'image_'+meta[f"image_{i}"] in ls_imTables:
                    printt('image_'+meta[f"image_{i}"])
                    # Specify the path to your .txt file
                    image_filename = 'image_'+meta[f"image_{i}"]
                    text_filename = image_filename.replace('.jpg', '.txt')

                    # Construct the path for the text file
                    text_path = f'/notebooks/DeployDATA/PPE_image_tables-extraction/{text_filename}'     

                    # Open the file and read its content
                    with open(text_path, 'r', encoding='utf-8') as file:
                        text_content = file.read()
                    
                    # Replace the table placeholders in the document
                    # printt('replacing with'+text_content, 'yellow')
                    doc = replace_first_image_placeholder(doc, text_content)
                    file.close()
                    chunk_images.append(PATH_IMAGES+'/tables/'+'image_'+meta[f"image_{i}"])
                    
                elif 'image_'+meta[f"image_{i}"] in ls_illustrations:
                    chunk_images.append(PATH_IMAGES+'/illustrations/'+'image_'+meta[f"image_{i}"])
                    printt('image_'+meta[f"image_{i}"], 'green')
                    # Specify the path to your .txt file
                    image_filename = 'image_'+meta[f"image_{i}"]
                    text_filename = image_filename.replace('.jpg', '')

                    text_content = image_description[text_filename]
                    xx = dc(text_content)

                    # printt('replacing with'+text_content, 'yellow')
                    # Replace the table placeholders in the document
                    
                    doc = replace_first_image_placeholder(doc, xx)
        

                i += 1
                if i >= max_tables:  # Limit to max_tables to avoid excessive looping
                    break
            
            images_to_plot.append(chunk_images)
        else:
            images_to_plot.append([None])
        printt('Chunk N°'+str(chunk_number+1),'yellow')
        print(meta['reference'])
        # Print the processed document
        print(doc)
#         encodeds = tokenizer_dragon(doc, return_tensors="pt", add_special_tokens=True).to('cuda')  # Tokenizes the prompt
#         # Print the length of the tokenized prompt
#         print(f"The prompt length is: {encodeds['input_ids'].size(1)}")
        
        # if encodeds['input_ids'].size(1)>50:
        chunks +='Reference: ' + meta['reference'] +'\n' +doc + '\n'
        chunks_list.append(meta['reference'] +'\n' +doc + '\n')
        # else:
        #     printt(doc, 'cyan')
    return chunks, chunks_list, images_to_plot,full_query_text 
embed_fn = DragonEmbeddingFunction()
collection = chroma_client.get_collection('FINAL_DATA', embedding_function=embed_fn)#Content_PPE_subchunked, Fairly_sectioned_db

def generate_answer(query,n_results=10,rerank_k=3,  model=model_agent, tokenizer=tokenizer_agent):
    # if tokenizer_dragon(query, return_tensors="pt", add_special_tokens=True).to('cuda')['input_ids'].size(1) <= 2:
    #     return 'Please, provide more context', None

    formatted_docs_retreived, chunks_list, images, full_query_text = process_documents_Wimages(collection, query_text=query, n_results=n_results)
    # full_query_text: is the new question+ history of Q&As
    
    if chunks_list ==[]:
        formatted_docs_retreived = 'No Relevant Context'
    else:
        formatted_docs_retreived, selected_docs, images_to_plot = reranker(full_query_text, contexts=chunks_list, images_to_plot_=images, k=rerank_k)#3
        if not selected_docs:
            formatted_docs_retreived = 'No Relevant Context'

    input_text = tokenizer.apply_chat_template([{"role": "user", "content": f"""{Main_Chatbot_Prompt}
    CONTEXT: {formatted_docs_retreived} QUESTION: {full_query_text}"""}], tokenize=False, add_generation_prompt=True)

    tokenizer_settings = {
        "padding": True,
        "truncation": True,
        "max_length": 4096,
        # "truncation_side": "left",  # Truncate from the beginning
    }
    encodeds = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, **tokenizer_settings).to('cuda')
    print(f"The prompt length is: {encodeds['input_ids'].size(1)}")

    streamer = TextStreamer(tokenizer, **tokenizer_settings, skip_prompt=True)
    with torch.no_grad():
        outputs = model.generate(input_ids=encodeds['input_ids'].to(model.device), max_new_tokens=4096, streamer=streamer)#4096
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return (generated_answer.split('\nmodel\n')[-1].strip(), images_to_plot)

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
import gradio as gr

def generate_pipeline(query, main_model=model_agent, main_tokenizer=tokenizer_agent):
    ARABIC =False
    
    if is_arabic(query):
        ARABIC =True
        print('Processing translation')
        query = translate_arabic_to_english(query)
        print('Finished translation: ', query)
        
    first_layer_response = generate_answer_reg_agent(query, reg_agent_model, reg_agent_tokenizer)

    if "APPROVED QUERY" in first_layer_response:
        RESULT = generate_answer(query, n_results=10, rerank_k=5 , model=main_model, tokenizer=main_tokenizer)    
        
        text_res = RESULT[0]
        if ARABIC:
            print('Processing translation')
            text_res = translate_english_to_arabic(RESULT[0])

        conversation_history.append(f"Assistant: {text_res}")
        print('images', RESULT[1])

        return '\n'.join(conversation_history), text_res, RESULT[1]
    
    else:
        return '\n'.join(conversation_history), first_layer_response, None
# Function to clear the conversation list
def clear_conv_list():
    global conversation_history, retrieval_conv_query_list
    conversation_history, retrieval_conv_query_list = [], []  # Clear the conversation list
    return "Conversation history cleared.", "",None

css = ".code-box {white-space: pre-wrap; word-wrap: break-word;  unicode-bidi: embed}"#direction: rtl;
js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# Gradio Interface
with gr.Blocks(theme=gr.themes.Base(), js=js_func, css=css) as interface:
    # Place the conversation history in a Textbox (to make it look like a box)
    # if conversation_history != []:
    #     history = gr.Code(label="Conversation History", value='\n'.join(conversation_history), language="markdown")
    # else:
    #     history = gr.Textbox(label="Conversation History", placeholder='No chat history')
    history = gr.Code(label="Conversation History", value='\n'.join(conversation_history), language="markdown")

    # Input for user query
    query = gr.Textbox(label="Query", placeholder="Ask a question here...")
    
    # Output for generated answer
    # answer_output_text = gr.Markdown(label="Generated Answer")
    answer_output_text = gr.Code(label="Generated Answer", value="", language="markdown", interactive=False, elem_classes="code-box")#, interactive=True, lines=10, show_label=True)
    #gr.Code(label="Generated Answer", value="", language="markdown", interactive=False, elem_classes="code-box")#, interactive=True, lines=10, show_label=True)

    # Output for generated plots
    answer_output_plot = gr.Gallery(label="Potential Relevant Illustrations")
    # Button to clear the conversation history
    clear_button = gr.Button("Clear Conversation History")
    
    # Trigger the pipeline to process the query and update the history
    query.submit(
        fn=generate_pipeline,
        inputs=query,
        outputs=[history, answer_output_text, answer_output_plot],
        show_progress=True
    )
    
    # Clear the conversation history when the button is clicked
    clear_button.click(
        fn=clear_conv_list,
        inputs=[],
        outputs=[history, answer_output_text, answer_output_plot]
    )

# Gradio Interface (if applicable)
def main():

    # Launch the Gradio interface
    interface.launch(share=True)
    
if __name__ == "__main__":
    main()
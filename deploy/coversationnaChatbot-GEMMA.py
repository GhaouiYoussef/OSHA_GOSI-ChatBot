import os
import torch
import json
import chromadb
from copy import deepcopy as dc
from transformers import AutoTokenizer, AutoModel
from chromadb import Documents, EmbeddingFunction, Embeddings
from termcolor import colored
from typing import List, cast
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig, TextStreamer
import gradio as gr

# Load environment variables from .env file
load_dotenv()
Reg_Agent_Prompt = os.getenv("Reg_Agent_Prompt")
Main_Chatbot_Prompt = '''You are an assistant specialized in providing accurate, concise, and contextually relevant information from the Occupational Safety and Health Handbook of KSA in the language of the questions asked. Your primary objective is to deliver organized, clear, and polite responses to user inquiries, incorporating basic physics and mathematics for necessary calculations. If the required information isn't available in the provided context, respectfully inform the user and refrain from guessing.
Instructions:
If the query is inst in english, try understanding it in english.
1. Comprehend the Query: Ensure a thorough understanding of the user's question. Identify key terms and any specific values provided, such as resistance (R) and current (I).
2. Review Chat History: Read the history chat between the assistant and the user to find if there is any relevance enough to make the LLM take previous chat into consideration.
3. Contextual Search: Search the context for relevant information or equations. For example, if the query mentions I and R, and the context discusses V, ensure that your calculation aligns with the relevant information (e.g., using Ohm’s Law: V = I * R).
4. Calculation Steps: If calculations are needed, perform them step by step, only after confirming that no direct information is available in the context.
5. Clear Explanation: Present each calculation step clearly and concisely, explaining how each value was derived and its relevance to the user’s question.
6. Verification: Cross-check the calculated results against the provided context to ensure accuracy and relevance.
7. Complete Response: Provide a comprehensive answer that integrates both the calculated results and information directly from the context.
8. Clarity and Brevity: Keep your response clear, concise, and free from unnecessary details to enhance user comprehension.
9. Source Attribution: Reference the specific sections or pages of the Handbook used for each piece of information provided.
10. Final Review: Conduct a final review of your response to ensure logical consistency, accuracy. Make sure references are always present. Make sure the question the answer is in the same language as the question.
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
cache_dir = './DeployDATA/my_local_cache'
#quantization_config=bnb_config,
model_id = "google/gemma-2-9b-it"#
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

# Load DPR models
query_encoder_dpr = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to('cuda')
context_encoder_dpr = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to('cuda')

# Load tokenizers
query_tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer_dpr = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')


device='cuda'
tokenizer_dragon = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
query_encoder = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder').to('cuda')
context_encoder = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder').to('cuda')

def printt(strr, color='red', attrs=['bold']):
    print(colored(strr, color, attrs=attrs))

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
    query_input = query_tokenizer_dpr(query, max_length=512, return_tensors='pt', padding=True, truncation=True).to(device)
    ctx_input = context_tokenizer_dpr(contexts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # Compute embeddings
    with torch.no_grad():
        query_emb = query_encoder_dpr(**query_input).pooler_output  # Use pooler_output for DPR
        ctx_emb = context_encoder_dpr(**ctx_input).pooler_output
        
    # Compute similarity scores using dot product
    scores_dpr = {i: (query_emb @ ctx_emb[i].T).item() for i in range(len(ctx_emb))}
    
    print('DPR Done')
    
    # Combining Dragon and dpr scores:
    scores = {i: scores_dpr[i] + scores_dragon[i] for i in range(len(ctx_emb))}
    print(scores)

    # Sort scores and select top documents
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print(sorted_scores)
    
    top = [(list(sorted_scores)[i][0], list(sorted_scores)[i][1]) for i in range(min(43, len(sorted_scores))) if list(sorted_scores)[i][1] >= RELEVENCE_THREASHOLD_Dragon+RELEVENCE_THREASHOLD_Dpr]#if list(sorted_scores)[i][1] >= RELEVENCE_THREASHOLD_Dragon

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
    full_query_text = '\n'.join(conversation_history) + '\n' + query_text
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
    
    return generated_answer.split('\nmodel\n')[-1].strip(), images_to_plot

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
        RESULT = generate_answer(query, n_results=50, model=main_model, tokenizer=main_tokenizer)    
        conversation_history.append(f"Assistant: {RESULT[0]}")
        return RESULT
    
    else:
        return first_layer_response, None

    # Modify the Gradio interface to include both text and image outputs
interface = gr.Interface(
    fn=generate_pipeline,
    inputs=[
        gr.Textbox(label="Query"),
        # gr.Slider(label="Number of chunks 1st rag", minimum=1, maximum=100, step=1, value=50),  # Slider for n_results
        # gr.Slider(label="Number of chunks Reranking", minimum=1, maximum=100, step=1, value=5)  # Slider for n_results
    ],
            outputs=[
        gr.Markdown(label="Generated Answer"),  # Changed to Markdown for bold text
        gr.Gallery(label="Generated Plots")  # Use Gallery to display multiple images
    ],
    title="OSHA AI Assistant",
    description="Ask a question related to the Occupational Safety and Health Handbook of KSA and get a detailed, contextual, and concise answer along with relevant plots."
)


# Gradio Interface (if applicable)
def main():

    # Launch the Gradio interface
    interface.launch(share=True)
    
if __name__ == "__main__":
    main()
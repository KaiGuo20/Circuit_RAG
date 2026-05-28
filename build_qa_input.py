import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
from typing import Callable

import openai
import json



from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import math
from typing import Dict, List, Tuple, Optional
import types
from accelerate import Accelerator
import os
import json
import pickle
import numpy as np
import torch
import networkx as nx
import numpy as np
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import openai
import re
# import networkx as nx
# from collections import defaultdict

# def extract_triples(reasoning_paths):
#     """Extract triples (subject, relation, object) from reasoning paths."""
#     triples = []
#     for path in reasoning_paths.split("\n"):
#         if "->" in path:
#             # Split the whole line by ->
#             parts = [p.strip() for p in path.split("->")]
#             # Traverse each triple in the path
#             for i in range(len(parts)-2):
#                 subject = parts[i]
#                 relation = parts[i+1]
#                 obj = parts[i+2]
#                 if subject and relation and obj:  # Ensure all parts are non-empty
#                     triples.append((subject, relation, obj))
#     return triples

# def build_relation_graph(triples):
#     """Build a relation graph with relations as nodes."""
#     G = nx.DiGraph()
#     for s, r, o in triples:
#         G.add_edge(s, r)
#         G.add_edge(r, o)
#     return G

# def run_personalized_pagerank(graph, personalized_nodes, alpha=0.85):
#     """Run Personalized PageRank (PPR)."""
#     if len(graph.nodes()) == 0:
#         return {}
#
#     personalization = {node: 0 for node in graph.nodes()}
#
#     # If no personalized nodes found, distribute weight evenly
#     if not any(node in personalization for node in personalized_nodes):
#         for node in personalization:
#             personalization[node] = 1.0 / len(personalization)
#     else:
#         # Assign weight to found personalized nodes
#         found_nodes = [node for node in personalized_nodes if node in personalization]
#         if found_nodes:
#             weight = 1.0 / len(found_nodes)
#             for node in found_nodes:
#                 personalization[node] = weight
#
#     try:
#         return nx.pagerank(graph, alpha=alpha, personalization=personalization)
#     except:
#         # If pagerank fails, return uniform distribution
#         return {node: 1.0/len(graph.nodes()) for node in graph.nodes()}

# def filter_reasoning_path(question, reasoning_paths, threshold=0.1):
#     """Filter reasoning paths based on PPR."""
#     if not reasoning_paths or not question:
#         return reasoning_paths
#
#     # Store original paths
#     original_paths = [path.strip() for path in reasoning_paths.split("\n") if path.strip()]
#
#     # Extract triples
#     triples = extract_triples(reasoning_paths)
#     if not triples:
#         return reasoning_paths
#
#     # Build relation graph
#     graph = build_relation_graph(triples)
#
#     # Extract keywords from question as personalized nodes
#     personalized_nodes = set()
#     for word in question.split():
#         if word in graph.nodes:
#             personalized_nodes.add(word)
#
#     # Run PPR
#     ppr_scores = run_personalized_pagerank(graph, personalized_nodes)
#
#     # If PPR fails or has no scores, return original paths
#     if not ppr_scores:
#         return reasoning_paths
#
#     # Compute score for each path and store endpoints
#     path_scores = []
#     path_endpoints = {}
#     for path in original_paths:
#         parts = [p.strip() for p in path.split("->")]
#         # Compute average score for path
#         path_score = sum(ppr_scores.get(part, 0) for part in parts) / len(parts)
#         start, end = parts[0], parts[-1]
#         path_scores.append((path_score, path))
#         path_endpoints[path] = (start, end)
#
#     # Sort by score
#     path_scores.sort(reverse=True)
#
#     # Deduplicate: for same start and end, keep only highest-scoring path
#     used_endpoints = set()
#     filtered_paths = []
#     for _, path in path_scores:
#         endpoints = path_endpoints[path]
#         if endpoints not in used_endpoints:
#             filtered_paths.append(path)
#             used_endpoints.add(endpoints)
#
#     # If no important paths found, return original paths
#     if not filtered_paths:
#         return reasoning_paths
#
#     return "\n".join(filtered_paths)

import networkx as nx
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_triples(reasoning_paths):
    """Extract triples (subject, relation, object) from reasoning paths."""
    triples = []
    for path in reasoning_paths.split("\n"):
        if "->" in path:
            parts = [p.strip() for p in path.split("->")]
            for i in range(len(parts) - 2):
                subject = parts[i]
                relation = parts[i + 1]
                obj = parts[i + 2]
                if subject and relation and obj:  # Ensure all parts are non-empty
                    triples.append((subject, relation, obj))
    return triples
def build_relation_graph(triples):
    """Build a relation graph with relations as nodes."""
    G = nx.DiGraph()
    for s, r, o in triples:
        G.add_edge(s, r)
        G.add_edge(r, o)
    return G


def run_personalized_pagerank(graph, personalized_nodes, alpha=0.85):
    """Run Personalized PageRank (PPR)."""
    if len(graph.nodes()) == 0:
        return {}

    personalization = {node: 0 for node in graph.nodes()}

    if not any(node in personalization for node in personalized_nodes):
        for node in personalization:
            personalization[node] = 1.0 / len(personalization)
    else:
        found_nodes = [node for node in personalized_nodes if node in personalization]
        if found_nodes:
            weight = 1.0 / len(found_nodes)
            for node in found_nodes:
                personalization[node] = weight

    try:
        return nx.pagerank(graph, alpha=alpha, personalization=personalization)
    except:
        return {node: 1.0 / len(graph.nodes()) for node in graph.nodes()}


# Load pretrained language model
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.eval()  # Set to evaluation mode
except Exception as e:
    print(f"Warning: Failed to load language model: {e}")
    model = None
    tokenizer = None

def evaluate_path_with_lm(question, path):
    """Evaluate path-question relevance using a language model."""
    if model is None or tokenizer is None:
        return 1.0  # If model loading failed, return default score
        
    try:
        with torch.no_grad():  # No gradient computation
            input_text = f"Question: {question} Path: {path}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            relevance_score = outputs.logits.softmax(dim=1)[0, 1].item()
        return relevance_score
    except Exception as e:
        print(f"Warning: LM evaluation failed: {e}")
        return 1.0  # Return default score on failure

def filter_reasoning_path(question, reasoning_paths, ppr_threshold=0.1, lm_threshold=0.5):
    """Filter reasoning paths based on PPR and language model."""
    if not reasoning_paths or not question:
        return reasoning_paths

    # Store original paths
    original_paths = [path.strip() for path in reasoning_paths.split("\n") if path.strip()]

    # Extract triples
    triples = extract_triples(reasoning_paths)
    if not triples:
        return reasoning_paths

    # Build relation graph
    graph = build_relation_graph(triples)

    # Extract keywords from question as personalized nodes
    personalized_nodes = set()
    for word in question.split():
        if word in graph.nodes:
            personalized_nodes.add(word)

    # Run PPR
    ppr_scores = run_personalized_pagerank(graph, personalized_nodes)

    # If PPR fails or has no scores, return original paths
    if not ppr_scores:
        return reasoning_paths

    # Compute combined score for each path and store endpoints
    path_scores = []
    path_endpoints = {}
    for path in original_paths:
        parts = [p.strip() for p in path.split("->")]
        # PPR score
        ppr_score = sum(ppr_scores.get(part, 0) for part in parts) / len(parts)

        # Language model score
        lm_score = evaluate_path_with_lm(question, path)

        # Only keep when both scores exceed threshold
        if ppr_score >= ppr_threshold and lm_score >= lm_threshold:
            combined_score = (ppr_score + lm_score) / 2  # average score
            start, end = parts[0], parts[-1]
            path_scores.append((combined_score, path))
            path_endpoints[path] = (start, end)

        # if lm_score >= lm_threshold:
        #     combined_score = lm_score # average score
        #     start, end = parts[0], parts[-1]
        #     path_scores.append((combined_score, path))
        #     path_endpoints[path] = (start, end)
    # Sort by score
    path_scores.sort(reverse=True)

    # Deduplicate: for same start and end, keep only the highest-scoring path
    used_endpoints = set()
    filtered_paths = []
    for _, path in path_scores:
        endpoints = path_endpoints[path]
        if endpoints not in used_endpoints:
            filtered_paths.append(path)
            used_endpoints.add(endpoints)

    # If no important paths found, return original paths
    if not filtered_paths:
        return reasoning_paths

    return "\n".join(filtered_paths)




# import numpy as np
# from sentence_transformers import SentenceTransformer, util

# # Load language model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def evaluate_with_lm(question, path):
#     embeddings = model.encode([question, path])
#     relevance_score = util.cos_sim(embeddings[0], embeddings[1]).item()
#     return relevance_score

# def filter_reasoning_path(question, reasoning_paths):
#     paths = reasoning_paths.strip().split("\n")
#     path_scores = []
#     for path in paths:
#         score = evaluate_with_lm(question, path)
#         path_scores.append((score, path))
#     # Sort by score
#     path_scores.sort(reverse=True, key=lambda x: x[0])
#     # Filter paths below threshold
#     threshold = np.percentile([ps[0] for ps in path_scores], 50)  # use median as threshold
#     filtered_paths = [ps[1] for ps in path_scores if ps[0] >= threshold]
#     return "\n".join(filtered_paths)





def path2text(context):
    openai.api_key = 'yours'

    # Call GPT-3.5 to convert context information into natural language sentences
    prompt = f"""
    Here are some reasoning paths separated by \n.
    {context}
    
    Please convert each reasoning path separated by \n into individual natural language sentences, without summarizing them into one sentence.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    # Get the generated natural language sentences
    generated_sentences = response['choices'][0]['message']['content']
    print("Generated Sentences:", generated_sentences)
    return generated_sentences

def save_sentences_to_json(sentences, index):
    # Dynamically generate filename
    filename = f'yours/expla_graph_text_{index}.json'

    # Ensure each generated sentence is a string and store in JSON format
    data = {
        "generated_sentences": sentences
    }

    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Sentences saved to {filename}")
import json
import os

def save_sentences_to_json_index(sentence, index):
    """
    Append a new sentence to a JSON file at the given index.
    """
    filename = f'yours/webqsp_text2.json'

    # # If file does not exist, initialize an empty list
    # if not os.path.exists(filename):
    #     with open(filename, "w") as f:
    #         json.dump([], f)

    # # Read existing JSON data
    # with open(filename, "r") as f:
    #     data = json.load(f)

    # # Insert new sentence at first line
    # data.insert(index, {"generated_sentence": sentence})

    # # Write back to JSON file
    # with open(filename, "w") as f:
    #     json.dump(data, f, indent=4)
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump([], f)

    # Read existing JSON data
    with open(filename, "r") as f:
        data = json.load(f)

    # Ensure list is long enough, pad with None if needed
    while len(data) <= index:
        data.append(None)

    # Update sentence at specified index
    data[index] = sentence

    # Write back to JSON file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Sentence saved to the index {index} in {filename}")

def read_sentence_from_index(filename, index):
    """
    Read sentence from the given index in a JSON file.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    # Check index range
    if index < len(data) and data[index] is not None:
        return data[index]
    else:
        raise IndexError(f"No data found at index {index}")


class PromptBuilder(object):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    # @staticmethod
    # def add_args(parser):
    #     parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path",
    #                         default='rmanluo/RoG')
    #     parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
    #     parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')

    def __init__(self, prompt_path, add_rule = False, use_true = False, cot = False, explain = False, use_random = False, each_line = False, maximun_token = 4096, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line
    #######################################
        # Add model-related attributes
        # self.maximun_token = 4096 - 100
        # self.model_path = 'rmanluo/RoG'
        # self.max_new_tokens = 512
        # self.dtype = 'fp16'

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_path,
        #     use_auth_token=True,
        #     use_fast=False
        # )

        # print("Loading model...")
        # # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7,0,1,2,3"
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_path,
        #     torch_dtype=self.DTYPE.get(self.dtype, None),
        #     device_map="auto"
        # )
    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    # SAQ_RULE_INSTRUCTION = """Please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""

    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""       
    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template
    
    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results
    
    def direct_answer(self, question_dict):
        graph = utils.build_graph(question_dict['graph'])
        entities = question_dict['q_entity']
        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction
    
    
    def process_input(self, question_dict,idx):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']
        
        if not question.endswith('?'):
            question += '?'
        

        if self.add_rule:
            graph = utils.build_graph(question_dict['graph'])
            entities = question_dict['q_entity']
            if self.use_true:
                rules = question_dict['ground_paths']
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict['predicted_paths']
            if len(rules) > 0:
                reasoning_paths = self.apply_rules(graph, rules, entities)
                lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]
                # print('lists_of_paths',lists_of_paths)

                # noise_paths = [
                #     'Elon Musk -> people.person.founded -> SpaceX', 
                #     'The Mona Lisa -> visual_art.artwork.displayed_at -> The Louvre', 
                #     'Black Hole -> physics.theory.predicted_by -> Albert Einstein', 
                #     'Amazon River -> geography.river.mouth -> Atlantic Ocean', 
                #     'Python -> programming_language.created_by -> Guido van Rossum', 
                #     'Eiffel Tower -> location.location_containedby -> Paris', 
                #     'Coca-Cola -> business.product.manufacturer -> The Coca-Cola Company', 
                #     'Mount Everest -> geography.mountain.elevation -> 8848 meters', 
                #     'Leonardo da Vinci -> people.person.known_for -> Mona Lisa', 
                #     'Bitcoin -> finance.currency.creator -> Satoshi Nakamoto', 
                #     'Tesla Model S -> automotive.model.manufacturer -> Tesla', 
                #     'Shakespeare -> book.author.written_works -> Hamlet', 
                #     'Jurassic Park -> film.film.director -> Steven Spielberg', 
                #     'Facebook -> business.company.founded_by -> Mark Zuckerberg', 
                #     'Great Wall of China -> location.location_length -> 21,196 km', 
                #     'The Beatles -> music.artist.genre -> Rock', 
                #     'Pluto -> astronomy.celestial_body.classification -> Dwarf planet', 
                #     'Harvard University -> education.university.founded_year -> 1636', 
                #     'Japan -> location.country.official_language -> Japanese', 
                #     'Olympic Games -> sports.sports_event.founded_year -> 1896', 
                #     'New York City -> location.city.population -> 8.5 million', 
                #     'Cristiano Ronaldo -> sports.athlete.plays_for -> Al-Nassr', 
                #     'Genghis Khan -> people.person.founded -> Mongol Empire', 
                #     'Machu Picchu -> location.historic_place.discovered_by -> Hiram Bingham', 
                #     'Google -> business.company.headquarters -> Mountain View, California', 
                #     'Netflix -> business.company.industry -> Streaming Services', 
                #     'The Moon -> astronomy.celestial_body.orbits -> Earth', 
                #     'Pacific Ocean -> geography.body_of_water.area -> 165.25 million km²', 
                #     'Albert Einstein -> people.person.theories -> Theory of Relativity', 
                #     'Wright Brothers -> transportation.aircraft.inventors -> Airplane'
                # ]

                # # Concatenate correct paths and noise paths
                # lists_of_paths = noise_paths + lists_of_paths
            else:
                lists_of_paths = []
            #input += self.GRAPH_CONTEXT.format(context = context)
            
        input = self.QUESTION.format(question = question)
        # MCQ
        if len(question_dict['choices']) > 0:
            choices = '\n'.join(question_dict['choices'])
            input += self.CHOICES.format(choices = choices)
            if self.add_rule:
                instruction = self.MCQ_RULE_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        # SAQ
        else:
            if self.add_rule:
                instruction = self.SAQ_RULE_INSTRUCTION
            else:
                instruction = self.SAQ_INSTRUCTION
        
        if self.cot:
            instruction += self.COT
        
        if self.explain:
            instruction += self.EXPLAIN
            
        if self.each_line:
            instruction += self.EACH_LINE
        
        if self.add_rule:
            other_prompt = self.prompt_template.format(instruction = instruction, input = self.GRAPH_CONTEXT.format(context = "") + input)
            context = self.check_prompt_length(other_prompt, lists_of_paths, self.maximun_token)
            print('idx', idx)
            # print('context--\n', context)
            # ##########path2text
            # description = path2text(context)
            # print('path2text\n',description)
            # save_sentences_to_json_index(description,idx)
            # PATH = 'yours'
            # filename = f"{PATH}/webqsp_text2.json"
            # # Load "generated_sentences" content from JSON file and concatenate into string
            # description = read_sentence_from_index(filename, idx)
            # print('context\n', description)
            # context = description


            # context = filter_reasoning_path(question, context)
            
            input = self.GRAPH_CONTEXT.format(context = context) + input   
        input = self.prompt_template.format(instruction = instruction, input = input)
        # print('input---------', input)
            
        return input
    
    def check_prompt_length(self, prompt, list_of_paths, maximun_token):
        '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
        original_path_count = len(list_of_paths)
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths
        else:
            print('too long------------')
            # Shuffle the paths
            random.shuffle(list_of_paths)
            new_list_of_paths = []
            # check the length of the prompt
            for p in list_of_paths:
                tmp_all_paths = "\n".join(new_list_of_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    final_path_count = len(new_list_of_paths)
                    paths_removed = original_path_count - final_path_count
                    print('too long------------')
                    print(f'Removed {paths_removed} paths: Original paths = {original_path_count}, New paths = {final_path_count}')
                    return "\n".join(new_list_of_paths)
                new_list_of_paths.append(p)


    def inference_single_path(self, path_hidden, question_hidden, classifier, scaler):
        """
        Inference function for a single path, returns probability instead of binary prediction
        """
        # Prepare features
        path_embedding = path_hidden.mean(dim=1).squeeze(0).cpu().detach().numpy()
        question_embedding = question_hidden.mean(dim=1).squeeze(0).cpu().detach().numpy()
        features = np.concatenate([path_embedding, question_embedding])
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get probability directly
        probability = classifier.predict_proba(features_scaled)[0][1]
        return probability, probability  # Return same value twice, first as score, second as probability


    def load_path_classifier(self, model_path='cwq_path_classifier_model.pkl'):
        """Load the path classifier model and scaler"""
        # if self.cached_classifier is None or self.cached_scaler is None:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.cached_classifier = model_data['classifier']
        self.cached_scaler = model_data['scaler']
        return self.cached_classifier, self.cached_scaler
    def predict_path_relevance(self,path_hidden, question_hidden):
        # Load saved model
        classifier, scaler = load_model()
        
        # Get prediction
        prediction, probability = inference_single_path(
            path_hidden, 
            question_hidden, 
            classifier, 
            scaler
        )
        
        return prediction, probability
    # def check_prompt_length(self, prompt, list_of_paths, maximun_token):
    #     '''Check whether the input prompt is too long. If it is too long, handle paths in batches.'''
    #     original_path_count = len(list_of_paths)
        
    #     # Find insertion position
    #     path_marker = "Reasoning Paths:"
    #     marker_pos = prompt.find(path_marker)
    #     question_pos = prompt.find("\n\nQuestion:")
        
    #     # Base prompt part
    #     prefix = prompt[:marker_pos + len(path_marker)] + "\n"
    #     suffix = prompt[question_pos:]
        
    #     # If original input does not exceed limit, return directly
    #     all_paths = "\n".join(list_of_paths)
    #     all_tokens = prefix + all_paths + suffix
        
    #     if self.tokenize(all_tokens) < maximun_token:
    #         return all_paths
            
    #     print('too long------------')
        
    #     # Split paths into multiple batches
    #     batches = []
    #     current_batch = []
    #     current_batch_tokens = len(self.tokenizer.encode(prefix + suffix))

    #     for path in list_of_paths:
    #         path_tokens = len(self.tokenizer.encode(path))
    #         if current_batch_tokens + path_tokens > maximun_token:
    #             # Current batch reached limit, save and start new batch
    #             if current_batch:
    #                 batches.append(current_batch)
    #             current_batch = [path]
    #             current_batch_tokens = len(self.tokenizer.encode(prefix + "\n" + path + suffix))
    #         else:
    #             current_batch.append(path)
    #             current_batch_tokens += path_tokens
        
    #     if current_batch:
    #         batches.append(current_batch)

    #     # Collect scores for all paths
    #     paths_with_scores = []
    #     classifier, scaler = self.load_path_classifier('yours')

    #     for batch in batches:
    #         # Build complete input for current batch
    #         batch_paths = "\n".join(batch)
    #         batch_tokens = prefix + batch_paths + suffix
            
    #         # Get hidden states for current batch
    #         inputs = self.tokenizer(batch_tokens, return_tensors="pt").to(self.model.device)
    #         outputs = self.model(**inputs, output_hidden_states=True)
    #         last_layer_hidden_state = outputs.hidden_states[len(outputs.hidden_states) // 2 + 2]

    #         # Get question hidden states (need to re-fetch for each batch since positions change)
    #         question_str = "\n\nQuestion:"
    #         question_start_idx = batch_tokens.find(question_str) + len(question_str)
    #         question_text = batch_tokens[question_start_idx:].strip()
    #         question_token_start = len(self.tokenizer.encode(batch_tokens[:question_start_idx], add_special_tokens=True)) - 1
    #         question_token_end = question_token_start + len(self.tokenizer.encode(question_text, add_special_tokens=False))
    #         question_hiddens = last_layer_hidden_state[:, question_token_start:question_token_end, :]

    #         # Process each path in current batch
    #         for path in batch:
    #             path_start = batch_tokens.find(path)
    #             if path_start != -1:
    #                 token_start = len(self.tokenizer.encode(batch_tokens[:path_start], add_special_tokens=True)) - 1
    #                 path_tokens = self.tokenizer.encode(path, add_special_tokens=False)
    #                 token_end = token_start + len(path_tokens)
    #                 path_hiddens = last_layer_hidden_state[:, token_start:token_end, :]
                    
    #                 prediction, probability = self.inference_single_path(path_hiddens, question_hiddens, classifier, scaler)
    #                 paths_with_scores.append((path, prediction, probability))
    #                 print(f"Path: {path} (prediction={prediction}, confidence: {probability:.4f})")

    #         # Manually clear GPU memory
    #         del inputs, outputs, last_layer_hidden_state
    #         torch.cuda.empty_cache()

    #     # Sort by prediction score and build final output
    #     paths_with_scores.sort(key=lambda x: x[1], reverse=True)
    #     final_paths = []
    #     current_tokens = len(self.tokenizer.encode(prefix + suffix))

    #     for path, pred, prob in paths_with_scores:
    #         path_str = f"{path}"
    #         path_tokens = len(self.tokenizer.encode(path_str))
            
    #         if current_tokens + path_tokens <= maximun_token:
    #             final_paths.append(path_str)
    #             current_tokens += path_tokens
    #         else:
    #             break
    #     print('final_path----', final_paths)
    #     return "\n".join(final_paths)


    # def check_prompt_length(self, prompt, list_of_paths, maximun_token):
    #     original_path_count = len(list_of_paths)
    #     all_paths = "\n".join(list_of_paths)
    #     all_tokens = prompt + all_paths

    #     # First check if it exceeds max length
    #     if self.tokenize(all_tokens) < maximun_token:
    #         return all_paths

    #     print('too long------------')
        
    #     # Find insertion position
    #     path_marker = "Reasoning Paths:"
    #     marker_pos = prompt.find(path_marker)
    #     question_pos = prompt.find("\n\nQuestion:")
        
    #     # Base prompt part
    #     prefix = prompt[:marker_pos + len(path_marker)] + "\n"
    #     suffix = prompt[question_pos:]
        
    #     # Compute base token count
    #     base_token_count = len(self.tokenizer.encode(prefix + suffix))
        
    #     # Set a smaller batch size limit (e.g., at most 500 tokens)
    #     MAX_BATCH_TOKENS = 500
        
    #     # Split paths into smaller batches
    #     batches = []
    #     current_batch = []
    #     current_batch_tokens = 0

    #     for path in list_of_paths:
    #         path_tokens = len(self.tokenizer.encode(path))
    #         if current_batch_tokens + path_tokens > MAX_BATCH_TOKENS:
    #             if current_batch:
    #                 batches.append(current_batch)
    #             current_batch = [path]
    #             current_batch_tokens = path_tokens
    #         else:
    #             current_batch.append(path)
    #             current_batch_tokens += path_tokens
        
    #     if current_batch:
    #         batches.append(current_batch)

    #     # Collect scores for all paths
    #     paths_with_scores = []
    #     classifier, scaler = self.load_path_classifier('yours')

    #     for batch_idx, batch in enumerate(batches):
    #         print(f"Processing batch {batch_idx + 1}/{len(batches)}")
    #         try:
    #             # Build complete input for current batch
    #             batch_paths = "\n".join(batch)
    #             batch_tokens = prefix + batch_paths + suffix
                
    #             # Get hidden states for current batch
    #             inputs = self.tokenizer(batch_tokens, return_tensors="pt").to(self.model.device)
    #             with torch.cuda.amp.autocast():  # use mixed precision
    #                 outputs = self.model(**inputs, output_hidden_states=True)
    #             last_layer_hidden_state = outputs.hidden_states[len(outputs.hidden_states) // 2 + 2]

    #             # Get question hidden states
    #             question_str = "\n\nQuestion:"
    #             question_start_idx = batch_tokens.find(question_str) + len(question_str)
    #             question_text = batch_tokens[question_start_idx:].strip()
    #             question_token_start = len(self.tokenizer.encode(batch_tokens[:question_start_idx], add_special_tokens=True)) - 1
    #             question_token_end = question_token_start + len(self.tokenizer.encode(question_text, add_special_tokens=False))
    #             question_hiddens = last_layer_hidden_state[:, question_token_start:question_token_end, :]

    #             # Process each path in current batch
    #             for path in batch:
    #                 path_start = batch_tokens.find(path)
    #                 if path_start != -1:
    #                     token_start = len(self.tokenizer.encode(batch_tokens[:path_start], add_special_tokens=True)) - 1
    #                     path_tokens = self.tokenizer.encode(path, add_special_tokens=False)
    #                     token_end = token_start + len(path_tokens)
    #                     path_hiddens = last_layer_hidden_state[:, token_start:token_end, :]
                        
    #                     prediction, probability = self.inference_single_path(path_hiddens, question_hiddens, classifier, scaler)
    #                     paths_with_scores.append((path, prediction, probability))

    #             # Immediately clear memory
    #             del inputs, outputs, last_layer_hidden_state, question_hiddens, path_hiddens
    #             torch.cuda.empty_cache()

    #         except RuntimeError as e:
    #             if "out of memory" in str(e):
    #                 print(f"OOM in batch {batch_idx + 1}, using default scores for remaining paths")
    #                 # Use default scores for remaining paths
    #                 for path in batch:
    #                     paths_with_scores.append((path, 0.5, 0.5))
                    
    #                 # Clear memory
    #                 torch.cuda.empty_cache()
    #             else:
    #                 raise e

    #     # Sort by prediction score and build final output
    #     paths_with_scores.sort(key=lambda x: x[1], reverse=True)
    #     final_paths = []
    #     current_tokens = base_token_count

    #     for path, pred, prob in paths_with_scores:
    #         path_str = f"{path}"
    #         path_tokens = len(self.tokenizer.encode(path_str))
            
    #         if current_tokens + path_tokens <= maximun_token:
    #             final_paths.append(path_str)
    #             current_tokens += path_tokens
    #         else:
    #             break

    #     return "\n".join(final_paths)
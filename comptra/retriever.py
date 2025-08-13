from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sklearn.metrics import pairwise_distances
from rank_bm25 import BM25Okapi
import numpy as np
import Stemmer
import bm25s
import os

from comptra.prompts.decompose import structural
from math import ceil

import spacy
nlp = spacy.load("en_core_web_sm")

from grakel import Graph

def build_graph(doc, verbose=False) :
    node_labels = {token.i : token.pos_ for token in doc}
    edges = {}
    edge_labels = {}
    for token in doc :
        for child in token.children :
            edges[(token.i, child.i)] = 1
            edge_labels[(token.i, child.i)] = child.dep_
    
    if verbose:
        print(f"node labels: {node_labels}")
        print(f"edge labels: {edge_labels}")
        print(f"edges: {edges}")

    G = Graph(
        edges,
        node_labels=node_labels,
        edge_labels=edge_labels
    )

    return G

from typing import List

def borda(
    list_of_indices_1: List[List[int]],
    list_of_indices_2: List[List[int]],
) -> List[List[int]]:
    """
    Merge 2 lists of indices into a single one with Borda Count.
    for all i, list_of_indices_1[i] and list_of_indices_2[i] are assumed not to have repeated elements.
    - list_of_indices_1: List[List[int]],
        First list of indices. Sorted from the least similar to the most similar.
    - list_of_indices_2: List[List[int]],
        Second list of indices. Sorted from the least similar to the most similar.
    """
    assert len(list_of_indices_1) == len(list_of_indices_2), f"{len(list_of_indices_1)} is not equal to {len(list_of_indices_2)}."
    list_of_indices = []
    for i in range(len(list_of_indices_1)):
        R1 = list_of_indices_1[i]
        R2 = list_of_indices_2[i]
        
        R1 = [int(element) for element in R1]
        R2 = [int(element) for element in R2]

        intersection = list(set(R1) & set(R2))
        # print(f"intersection: {intersection}")
        fusion = {}
        # Start by R2 because we assume R1's indices are best overall
        for j in range(len(R2)):
            # Give an advantage to those who are in both sets
            if R2[j] in intersection:
                fusion[R2[j]] = (j + 1) + 1.5
            else:
                fusion[R2[j]] = (j + 1)
        # print(f"first step: {fusion}")
        for j in range(len(R1)):
            if R1[j] in fusion:
                # That means R1[j] belongs to R2
                # Thus it is in `intersection`
                # It already received the bonus in the above loop
                fusion[R1[j]] += (j + 1)
            else:
                fusion[R1[j]] = (j + 1)
        # print(f"second step: {fusion}")
        L = [(key, v) for (key, v) in fusion.items()]
        L = [(key, v, j) for j, (key, v) in enumerate(L)]
        L = sorted(L, key=lambda x: (x[1], x[2]))
        list_of_indices.append([a for (a, _, _) in L])
    return list_of_indices

from comptra.utils import lcs
from comptra.data.dataset import get_datasets
import multiprocess as mp

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Retriever:
    """
    Class which defines a Retriever
    """
    def __init__(
        self,
        source_language: str = "English",
        dataset_name_or_path: str = "flores",
        retriever_type: str = "bm25s",
        target_language: str = "French",
        variant: str = "robertson",
        ds_src = None,
        ds_tgt = None,
        seed = 122,
        path = None
    ) -> None:
        
        self.ds_src = ds_src if ds_src else get_datasets(dataset_name_or_path, source_language)
        self.ds_tgt = ds_tgt if ds_tgt else get_datasets(dataset_name_or_path, target_language)
        self.retriever_type = retriever_type
        
        if retriever_type == "SONAR":
            try:
                self.embedder = TextToEmbeddingModelPipeline(
                    encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
                )
            except Exception as e:
                print(f"`TextToEmbeddingModelPipeline` raises the following exception: '{e}'")
            try:
                self.X_src_devtest = np.fromfile(
                    os.path.join(
                        os.path.join(os.path.dirname(__file__), "data", dataset_name_or_path),
                        "eng/SONAR/devtest.bin",
                    ),
                    dtype=np.float32,
                    count=-1,
                ).reshape(len(self.ds_src["devtest"]), -1)
            except Exception as _:
                pass
            
            self.X_src_dev = np.fromfile(
                os.path.join(
                    os.path.join(os.path.dirname(__file__), "data", dataset_name_or_path),
                    "eng/SONAR/dev.bin",
                ),
                dtype=np.float32,
                count=-1,
            ).reshape(len(self.ds_src["dev"]), -1)
        
            if path and os.path.exists(os.path.join(path, "translate_1.bin")):
                import json
                self.X = np.fromfile(os.path.join(path, "translate_1.bin"), dtype=np.float32, count=-1).reshape(-1, 1024)
                L = []
                with open(
                    os.path.join(path, "divide_1.jsonl"), "r"
                ) as fin:
                    for line in fin:
                        L.extend(
                            json.loads(line)["propositions"]
                        )
                self.L = L
                print(f"Length: {len(self.L)}. Shape: {self.X.shape}")

        elif "lcs" in retriever_type.lower():
            self.tokenized_corpus = [example["sentence"].split(" ") for example in self.ds_src["dev"]]
        
        elif "grakel" in retriever_type.lower():
            from grakel.kernels import VertexHistogram, EdgeHistogram
            self.vertex_kernel = VertexHistogram(n_jobs=-1)
            self.edge_kernel = EdgeHistogram(n_jobs=-1)

            self.G_dev = [build_graph(nlp(sentence)) for sentence in self.ds_src["dev"]["sentence"]]
            # self.G_devtest = [build_graph(nlp(sentence)) for sentence in self.ds_src["devtest"]["sentence"]]
        
        elif "bm25s" in retriever_type:
            corpus = [example["sentence"] for example in self.ds_src["dev"]]
            # optional: create a stemmer
            # stemmer = Stemmer.Stemmer("english")
            try:
                stemmer = Stemmer.Stemmer(source_language.lower())
            except Exception as e:
                print(f"Exception for Stemmer: {e}")
                stemmer = Stemmer.Stemmer("english")
            # tokenize the corpus and only keep the ids (faster and saves memory)
            retriever = bm25s.BM25(
                method="robertson" if variant is None else variant,
                k1=1.5,
                b=0.7,
                delta=1.5 if variant == "bm25+" else None,
            )
            corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
            # corpus_tokens = bm25s.tokenize(corpus, stopwords=source_language[:2].lower(), stemmer=stemmer)
            retriever.index(corpus_tokens)
            self.retriever = retriever
            self.stemmer = stemmer

        elif "bm25" in retriever_type:
            corpus = [example["sentence"] for example in self.ds_src["dev"]]
            bm25 = BM25Okapi([sentence.split(" ") for sentence in corpus])
            self.retriever = bm25
        
        elif "Random" in retriever_type:
            self.rng = np.random.default_rng(seed)
        else:
            raise ValueError(f"{retriever_type} is not supported!")

    def query(
        self,
        sentence: str,
        k: int,
        idx_sentence: int = None,
        level: int = None
    ):
        """
        """
        if self.retriever_type == "SONAR":
            try:
                if level == 0:
                    print(f"---\nA: {sentence}\nB: {self.ds_src['devtest'][idx_sentence]['sentence']}\n---")
                    emb = self.X_src_devtest[idx_sentence].reshape(1, -1)
                elif level == 1:
                    print(f"---\nA: {sentence}\nB: {self.L[idx_sentence]}\n---")
                    emb = self.X[idx_sentence].reshape(1, -1)
                else:
                    raise ValueError(f"Unsupported level: {level}")
            except Exception as e:
                emb = self.embedder.predict([sentence], source_lang="eng_Latn")
                emb = emb.detach().numpy().reshape(1, -1) 
            # get the cosine similarity matrix
            D = 1 - pairwise_distances(emb, self.X_src_dev, metric="cosine")
            indices = np.argsort(D.reshape(-1))[-k:] # Least similar to most similar
            indices = [int(element) for element in indices]
            demonstrations = [
                (self.ds_src["dev"][i]["sentence"], self.ds_tgt["dev"][i]["sentence"])
                for i in indices
            ]
            return demonstrations
        
        elif "grakel" in self.retriever_type.lower():
            G_devtest = [build_graph(nlp(sentence))]
            K_vertex = self.vertex_kernel.fit(self.G_dev).transform(
                G_devtest
            )
            K_edge = self.edge_kernel.fit(self.G_dev).transform(
                G_devtest
            )
            indices_1 = list(K_vertex.argsort(-1)[0, -k:])
            indices_2 = list(K_edge.argsort(-1)[0, -k:])
            indices = borda([indices_1], [indices_2])[0][-k:]
            demonstrations = [
                (self.ds_src["dev"][i]["sentence"], self.ds_tgt["dev"][i]["sentence"])
                for i in indices
            ]
            return demonstrations
        elif "lcs" in self.retriever_type.lower():
            # lcs_values = [lcs(candidate, sentence.split(" ")) for candidate in self.tokenized_corpus]
            def f(candidate):
                return lcs(candidate, sentence.split(" "))
            
            p = mp.Pool(8)
            lcs_values = p.map(f, self.tokenized_corpus)
            # lcs_values = [f(candidate) for candidate in self.tokenized_corpus]
            indices = np.argsort(np.array(lcs_values))[-k:].tolist()
            indices = [int(element) for element in indices]
            demonstrations = [
                (self.ds_src["dev"][i]["sentence"], self.ds_tgt["dev"][i]["sentence"])
                for i in indices
            ]
            return demonstrations

        elif "structural" in self.retriever_type:
            subparts = structural(sentence, -1)
            sub = ceil(k / len(subparts)) * 2 # reduce overlap
            indices = []
            list_of_indices = []
            for subpart in subparts:
                query_tokens = bm25s.tokenize(subpart, stemmer=self.stemmer)
                results, scores = self.retriever.retrieve(query_tokens, corpus=None, k=sub, show_progress = False)
                subpart_indices = list(results[0])[::-1] # Least to most similar
                list_of_indices.append(subpart_indices)
            for element in zip(*list_of_indices):
                indices.extend(element)
            indices = list(set([int(idx) for idx in indices]))
            demonstrations = [
                (self.ds_src["dev"][i]["sentence"], self.ds_tgt["dev"][i]["sentence"])
                for i in indices[-k:]
            ]
            return demonstrations
        
        elif "bm25s" in self.retriever_type:
            query_tokens = bm25s.tokenize(sentence, stemmer=self.stemmer)
            results, scores = self.retriever.retrieve(query_tokens, corpus=None, k=k, show_progress = False)
            indices = list(results[0])[::-1]  # Least similar to most similar
            indices = [int(element) for element in indices]
            demonstrations = [
                (self.ds_src["dev"][i]["sentence"], self.ds_tgt["dev"][i]["sentence"])
                for i in indices
            ]
            return demonstrations
        
        elif "bm25" in self.retriever_type:
            scores = self.retriever.get_scores(sentence.split(" "))
            scores = list(scores)
            indices = np.argsort(scores)[-k:]
            demonstrations = [
                (self.ds_src["dev"][i]["sentence"], self.ds_tgt["dev"][i]["sentence"])
                for i in indices
            ]
            return demonstrations
        
        elif "Random" in self.retriever_type:
            indices = self.rng.choice(len(self.ds_src["dev"]), size=k, replace=False).tolist()
            demonstrations = [
                (self.ds_src["dev"][i]["sentence"], self.ds_tgt["dev"][i]["sentence"])
                for i in indices
            ]
            return demonstrations
        else:
            pass

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    retriever = Retriever(
        retriever_type = "bm25s",
        source_language = "English",
        target_language = "French",
        dataset_name_or_path = "flores"
    )
    print(retriever.query("I want to eat your pancreas.", 5))
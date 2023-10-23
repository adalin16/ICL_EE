import numpy as np
import pandas as pd
import torch
import openai
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as sf1_score
from openai.embeddings_utils import get_embedding
from easyinstruct import (
    BasePrompt,
    IEPrompt,
    ZeroshotCoTPrompt,
    FewshotCoTPrompt,
    BatchPrompt,
)
from easyinstruct.utils.api import set_openai_key, set_anthropic_key, set_proxy

set_openai_key("OPEN_AI.KEY")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def chunks(lst, n):
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def decode(embed_type, tok, model, corpus):
    embeddings = []

    for corpus_tmp in tqdm(chunks(corpus, 32)):
        encoding = tok.batch_encode_plus(corpus_tmp, padding=True, truncation=True)
        sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
        sentence_batch, attn_mask = torch.LongTensor(sentence_batch).to(
            device
        ), torch.LongTensor(attn_mask).to(device)

        with torch.no_grad():
            embedding_output_batch = model(sentence_batch, attn_mask)
            if embed_type == "mean":
                sentence_embeddings = mean_pooling(embedding_output_batch, attn_mask)
            elif embed_type == "CLS":
                sentence_embeddings = embedding_output_batch[0][:, 0, :]
        embeddings.append(sentence_embeddings.detach().cpu().numpy())

        del sentence_batch, attn_mask, embedding_output_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0)


def perplexity(tok, model, corpus):
    loss = []
    for corpus_tmp in tqdm(chunks(corpus, 32)):
        encoding = tok.batch_encode_plus(corpus_tmp, padding=True, truncation=True)
        sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
        sentence_batch, attn_mask = torch.LongTensor(sentence_batch).to(
            device
        ), torch.LongTensor(attn_mask).to(device)

        with torch.no_grad():
            loss_batch = model(input_ids=sentence_batch, labels=sentence_batch)
        loss.append(torch.exp(loss_batch.loss))
    return loss


def API_decode(corpus, embedding_model, data):
    embedding = []
    for sentence in corpus:
        embedding.append(get_embedding(sentence, engine=embedding_model))

    embedding_df = pd.DataFrame(embedding)
    embedding_df.to_csv(data + "_api.csv")
    return np.concatenate(embedding, axis=0)


def prediction(
    test_sentences,
    train_sentences,
    train_sentities,
    icl_data,
    num_neighbors,
):
    pred_entites = []
    for i, sentence in enumerate(test_sentences[0:100]):
        ieprompts = IEPrompt(task="ner")
        samples = []
        for j in range(num_neighbors):
            train_sentence = icl_data["kNN_dev_train"][j][0]
            index = train_sentences.index(train_sentence)

            sample = {}
            sample["Input"] = train_sentence
            out = []
            for ent in train_entities[index]:
                a = {}
                a["E"] = ent[1]
                a["W"] = ent[0]
                out.append(a)
            sample["Output"] = str(out)
            samples.append(sample)
        ieprompts.build_prompt(prompt=sentence, labels=labels, examples=samples)
        pred_entites.append(ieprompts.get_openai_result(engine="gpt-3.5-turbo-0301"))
    return pred_entities


def token_level_evaluation(final_pred, test_labels, dataprocessor):
    final_prediction, final_ground = [], []
    for f, t in zip(final_pred, test_labels):
        if len(f) != len(t):
            print(len(f), len(t))
            print(f, t)
        else:
            final_prediction.extend(dataprocessor.tag2bio(f))
            final_ground.extend(dataprocessor.tag2bio(t))
    return f1_score(final_ground, final_prediction, average="macro")


def entity_level_evaluation(final_pred, test_labels, dataprocessor):
    final_prediction, final_ground = [], []
    for f, t in zip(final_pred, test_labels):
        if len(f) != len(t):
            print(len(f), len(t))
        else:
            final_prediction.append(dataprocessor.tag2bio(f))
            final_ground.append(dataprocessor.tag2bio(t))
    return sf1_score(final_ground, final_prediction, average="macro")


def evaluation(test_sentences, test_labels, pred_entities, dataprocessor):
    final_pred = []
    for sentence, pred_item in zip(test_sentences, pred_entites):
        sentence = sentence.lower().split()
        label_list = ["O"] * len(sentence)
        if not pred_item.startswith("["):
            print(sentence)
        elif len(pred_item) > 2:
            try:
                final_dictionary = eval(pred_item[1:-1])

                label = []
                words = []
                for item in final_dictionary:
                    if isinstance(item, dict):
                        if item["E"] in labels:
                            word = item["W"].lower().split()
                            words.extend(word)
                            label.extend([item["E"]] * len(word))
                if len(words) > 0:
                    for w, l in zip(words, label):
                        try:
                            label_list[sentence.index(w)] = l
                        except:
                            print(sentence, w + "---" + l)
            except:
                print(sentence)
        final_pred.append(label_list)

    token_level = token_level_evaluation(final_pred, test_labels, dataprocessor)
    entity_level = entity_level_evaluation(final_pred, test_labels, dataprocessor)
    return token_level, entity_level

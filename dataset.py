import os
import sys
import json
from glob import glob
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from torch.utils.data import Dataset, DataLoader, TensorDataset


class DataProcessor:
    def __init__(self, labels, task, train_docs=None):
        # Initialize the DataProcessor class with labels, task, and optional train_docs.
        self.labels = self.generate_labels(labels)
        self.label2id = {label: id for id, label in enumerate(self.labels)}
        self.id2label = {id: label for label, id in self.label2id.items()}
        self.task = task
        if self.task == "measeval":
            train_texts = " ".join(
                [
                    open(txt_file, "r", encoding="utf8").read()
                    for txt_file in glob(os.path.join(train_docs, "*.txt"))
                ]
            )
            trainer = PunktTrainer()
            trainer.INCLUDE_ALL_COLLOCS = True
            trainer.train(train_texts)

            self.sentenizer = PunktSentenceTokenizer(trainer.get_params())
            self.tokenizer = RegexpTokenizer(r"\w+|\(|\)|\[|\]|[-–.,]|\S+")

    def token_in_span_stem(self, token, start, end):

        if token[1] >= start and token[2] <= end:
            return True
        else:
            return False

    def token_in_span_1(self, token, start, end):

        if token[1] >= start and token[2] <= end:
            return True
        else:
            return False

    def tag2bio(self, labels):
        bio_labels = []
        prev_label = labels[0]
        if prev_label != "O":
            bio_labels.append("B-" + prev_label)
        else:
            bio_labels.append(prev_label)

        for label in labels[1::]:
            if label == prev_label and prev_label != "O":
                bio_labels.append("I-" + label)
            elif label != prev_label and label != "O":
                bio_labels.append("B-" + label)
            else:
                bio_labels.append(label)
            prev_label = label

        id_labels = [self.label2id[l] for l in bio_labels]
        return id_labels

    def generate_labels(self, labels):
        bio_labels = []
        for label in labels:
            bio_labels.append("B-" + label)
            bio_labels.append("I-" + label)
        bio_labels.append("O")
        return bio_labels

    def get_dataframes(self, docs_dir, annotations_dir, part):
        tsv_files = (
            glob(os.path.join(annotations_dir, "*.tsv"))
            if annotations_dir is not None
            else []
        )
        txt_files = glob(os.path.join(docs_dir, "*.txt"))

        from collections import namedtuple

        Sentence = namedtuple("Sentence", ["start", "end", "text"])

        docs = {"docId": [], "text": []}
        for txt_file in txt_files:
            docs["text"].append(open(txt_file, "r", encoding="utf8").read())
            docs["docId"].append(txt_file.split("/")[-1].split(".")[0])
        docs = pd.DataFrame(docs)

        docs.loc[:, "texts"] = docs.text.apply(
            lambda x: [
                Sentence(start, end, text)
                for (start, end), text in zip(
                    self.sentenizer.span_tokenize(x), self.sentenizer.tokenize(x)
                )
            ]
        )

        docs = (
            docs.set_index(["docId", "text"])["texts"]
            .apply(pd.Series)
            .stack()
            .reset_index()
            .drop(["level_2", "text"], axis=1)
            .rename(columns={0: "text"})
        )

        if annotations_dir is not None:
            frames = []
            for tsv_file in tsv_files:
                frames.append(pd.read_csv(tsv_file, sep="\t"))
            annotations = pd.concat(frames, ignore_index=True)
            annotations = annotations.replace(np.nan, "{}")
            annotations.loc[:, "other"] = annotations.other.apply(lambda x: eval(x))
        else:
            annotations = pd.DataFrame(
                columns=[
                    "docId",
                    "annotSet",
                    "annotType",
                    "startOffset",
                    "endOffset",
                    "annotId",
                    "text",
                    "other",
                ]
            )

        corqualifier = False
        if corqualifier:
            corrQualifiers = {}
            for row in annotations[annotations.annotType == "Qualifier"].itertuples():
                docId, start, end, qualifies = (
                    row.docId,
                    row.startOffset,
                    row.endOffset,
                    row.other["Qualifies"],
                )
                annotType = annotations[
                    (annotations.docId == docId) & (annotations.annotId == qualifies)
                ].annotType.values[0]
                corrQualifiers[(start, end, docId)] = f"{annotType}Qualifier"
            annotations.loc[:, "annotType"] = annotations.apply(
                lambda r: r.annotType
                if r.annotType != "Qualifier"
                else corrQualifiers[(r.startOffset, r.endOffset, r.docId)],
                axis=1,
            )

        return docs, annotations

    def tokenize_sent_with_spans(self, sent):
        from collections import namedtuple

        Token = namedtuple("Token", ["start", "end", "text"])
        tokens = [
            Token(sent.start + start, sent.start + end, token)
            for (start, end), token in zip(
                self.tokenizer.span_tokenize(sent.text),
                self.tokenizer.tokenize(sent.text),
            )
        ]
        return tokens

    def add_annotations_to_tokens(self, tokens, labels):
        AnnotatedToken = namedtuple("AnnotatedToken", ["start", "end", "text", "label"])
        return [
            AnnotatedToken(tok.start, tok.end, tok.text, label)
            for tok, label in zip(tokens, labels)
        ]

    def add_entities(self, tokens):
        AnnotatedToken = namedtuple("AnnotatedToken", ["start", "end", "text"])
        return [AnnotatedToken(tok[0], tok[1], tok[2]) for tok in tokens]

    def token_in_span_measeval(self, token, start, end):
        if (
            start <= token.start < token.end <= end
            or start <= token.start < end <= token.end
            or token.start <= start < token.end <= end
            or token.start <= start <= end <= token.end
        ):
            return True
        else:
            return False

    def create_multitask_examples(self, docs_dir, annots_dir, part="train"):
        docs, annots = self.get_dataframes(docs_dir, annots_dir, part=part)
        docs = docs[docs.docId.isin(annots.docId.unique())]
        Example = namedtuple(
            "Example", ["docId", "quantity", "tokens", "mods", "units", "sent"]
        )

        examples = []
        spans_types = [
            "MeasuredEntityQualifier",
            "MeasuredPropertyQualifier",
            "QuantityQualifier",
            "MeasuredEntity",
            "MeasuredProperty",
            "Quantity",
        ]
        sents = defaultdict(list)
        for docId in annots.docId.unique():
            doc_annots = annots[annots.docId == docId]
            doc_texts = docs[docs.docId == docId]

            for annotSet in doc_annots.annotSet.unique():
                quant_annots = doc_annots[doc_annots.annotSet == annotSet]
                other = quant_annots[quant_annots.annotType == "Quantity"].other.values[
                    0
                ]
                quant_start = quant_annots[
                    quant_annots.annotType == "Quantity"
                ].startOffset.min()
                quant_end = quant_annots[
                    quant_annots.annotType == "Quantity"
                ].endOffset.max()

                contexts = doc_texts[
                    doc_texts.text.apply(
                        lambda x: x.start <= quant_start <= quant_end <= x.end
                    )
                ]
                for sent in contexts.itertuples():
                    tokens = self.tokenize_sent_with_spans(sent.text)
                    example = {}
                    labels = {}
                    entity = []
                    for annotType in spans_types:
                        if annotType == "Qualifier":
                            annot_start = quant_annots[
                                quant_annots.annotType == annotType
                            ].startOffset.values[0]
                            annot_end = quant_annots[
                                quant_annots.annotType == annotType
                            ].endOffset.values[0]
                        elif annotType == "Quantity":
                            annot_start = quant_annots[
                                quant_annots.annotType == annotType
                            ].startOffset.min()
                            annot_end = quant_annots[
                                quant_annots.annotType == annotType
                            ].endOffset.max()
                            entity.append(
                                (
                                    annot_start,
                                    annot_end,
                                    sent.text.text[
                                        annot_start
                                        - sent.text.start : annot_end
                                        - sent.text.start
                                    ],
                                )
                            )
                            sents[sent.text.text].append(
                                (tokens, annot_start, annot_end)
                            )
                        else:
                            annot_start = quant_annots[
                                quant_annots.annotType == annotType
                            ].startOffset.min()
                            annot_end = quant_annots[
                                quant_annots.annotType == annotType
                            ].endOffset.max()
                        if annotType not in quant_annots.annotType.values:
                            labels[annotType] = ["O"] * len(tokens)
                        else:
                            labels[annotType] = [
                                annotType
                                if self.token_in_span_measeval(
                                    token, annot_start, annot_end
                                )
                                else "O"
                                for token in tokens
                            ]
                    if "mods" in other:
                        example["mods"] = "+".join(other["mods"])
                    else:
                        example["mods"] = "O"

                    unit = ""
                    if "unit" in other:
                        unit = other["unit"]

                    if set(labels["Quantity"]) == {"O"}:
                        continue

                    example["tokens"] = self.add_annotations_to_tokens(
                        tokens,
                        [
                            "+".join(labs)
                            for labs in zip(
                                *[labels[annotType] for annotType in spans_types]
                            )
                        ],
                    )
                    example["quantity"] = self.add_entities(entity)
                    example["docId"] = docId
                    example["units"] = unit
                    example["sent"] = sent.text
                    examples.append(
                        Example(
                            example["docId"],
                            example["quantity"],
                            example["tokens"],
                            example["mods"],
                            example["units"],
                            example["sent"],
                        )
                    )

        return examples, sents

    def get_data_measeval(self, docs_dir, annots_dir, part="train"):
        examples, sents = self.create_multitask_examples(docs_dir, annots_dir, part)
        sentences, labels, ners = [], [], []
        for item in sents.keys():
            entity = []
            if len(sents[item]) > 0:
                tokens = sents[item][0][0]
                sent = [token.text for token in tokens]
                quantity = ["O"] * len(tokens)
                for values in sents[item]:
                    annot_start, annot_end = values[1:3]
                    sentence = item
                    entity.append((sentence[annot_start:annot_end], "Quantity"))
                    quantity = [
                        "Quantity"
                        if self.token_in_span_measeval(token, annot_start, annot_end)
                        else quantity[index]
                        for index, token in enumerate(tokens)
                    ]

                sentences.append(" ".join(sent).strip())
                labels.append(quantity)
                ners.append(entity)
        return sentences, labels, ners

    def get_data_scierc(self, file_path):
        with open(file_path) as file:
            files = file.readlines()

        sentences, labels, ners = [], [], []
        for line in files:
            line = json.loads(line)
            for j in line["sentences"]:
                sentences.append(" ".join(j))
            ners.extend(line["ner"])
            start = 0
            for sentence, ner in zip(line["sentences"], line["ner"]):
                label = ["O"] * len(sentence)
                for item in ner:
                    for i in range(item[0], item[1] + 1):
                        label[i - start] = item[2]
                start += len(label)
                labels.append(self.tag2bio(label))
        return sentences, labels, ners

    def get_data_ade(self, labels):
        with open(file_path) as file:
            files = file.readlines()

        sentences, labels, ners = [], [], []
        for line in files:
            all_data = json.loads(line)
            for a in all_data:
                sentences.append(" ".join(a["tokens"]).strip())

                label = ["O"] * len(a["tokens"])
                ner = a["entities"]
                for ner_item in ner:
                    for i in range(int(ner_item["start"]), int(ner_item["end"])):
                        label[i] = ner_item["type"]
                labels.append(label)
                ners.append(ner)
        return sentences, labels, ners

    def get_data_stem(self, file_path):
        sentences = []
        labels = []
        entities = []
        for subdir, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        subdir + "/" + file, errors="ignore", encoding="utf8"
                    ) as annotfile:
                        annots = annotfile.read().splitlines()
                    ann_file = file[:-4] + ".ann"
                    df = pd.read_csv(
                        subdir + "/" + ann_file,
                        sep="^([^\s]*)\s",
                        engine="python",
                        header=None,
                    ).drop(0, axis=1)
                    ner = []
                    for item in df[2]:
                        values = item.split("\t")
                        val = values[0].split()
                        ner.append((val[0], int(val[1]), int(val[2]), values[1]))

                    start_ = 0
                    start_1 = 0
                    for annot in annots:

                        label = ["O"] * len(annot.split())
                        sentence = []
                        entity = []
                        for ner_item in ner:
                            if ner_item[0] not in [
                                "Task",
                                "Object",
                                "Result",
                            ] and self.token_in_span_stem(
                                (ner_item[3], start_, start_ + len(annot)),
                                ner_item[1],
                                ner_item[2],
                            ):

                                entity.append((ner_item[1], ner_item[2], ner_item[0]))
                        start_ += len(annot) + 1
                        entities.append(entity)
                        for i, token in enumerate(annot.split()):
                            for ner_item in ner:
                                if ner_item[0] not in [
                                    "Task",
                                    "Object",
                                    "Result",
                                ] and self.token_in_span_1(
                                    (token, start_1, len(token) + start_1),
                                    ner_item[1],
                                    ner_item[2],
                                ):
                                    label[i] = ner_item[0]

                            start_1 += len(token) + 1
                            sentence.append(token)

                        sentences.append(sentence)
                        labels.append(label)

        final_sentences, final_labels, final_ners = [], [], []
        for sentence, label, entity in zip(sentences, labels, entities):
            if len(sentence) > 2:
                final_sentences.append(" ".join(sentence).strip())
                final_labels.append(label)
                final_ners.append(entity)
        return final_sentences, final_labels, final_ners

    def get_data_wlpc(self, file_path):
        sentences, labels, ners = [], [], []

        for subdir, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        subdir + "/" + file, errors="ignore", encoding="utf8"
                    ) as annotfile:
                        annots = annotfile.read().splitlines()
                    ann_file = file[:-4] + ".ann"
                    df = pd.read_csv(
                        subdir + "/" + ann_file,
                        sep="^([^\s]*)\s",
                        engine="python",
                        header=None,
                    ).drop(0, axis=1)
                    ner = []
                    for type_item, item in zip(df[1], df[2]):
                        if type_item.startswith("T"):
                            values = item.split("\t")
                            val = values[0].split()
                            if ";" in val[2]:
                                ner.append(
                                    (val[0], int(val[1]), int(val[3]), values[1])
                                )
                            else:
                                ner.append(
                                    (val[0], int(val[1]), int(val[2]), values[1])
                                )

                    start = 0
                    for annot in annots:

                        label = ["O"] * len(annot.split())
                        sentence, ner = [], []
                        for ner_item in ner:
                            if ner_item[0] not in [
                                "Misc",
                                "Unit",
                            ] and self.token_in_span_stem(
                                (ner_item[3], start_, start_ + len(annot)),
                                ner_item[1],
                                ner_item[2],
                            ):
                                ner.append((ner_item[3], ner_item[0]))

                        ners.append(ner)

                        for i, token in enumerate(annot.split()):

                            for ner_item in ner:
                                if ner_item[0] not in [
                                    "Misc",
                                    "Unit",
                                ] and self.token_in_span_stem(
                                    (token, start, len(token) + start),
                                    ner_item[1],
                                    ner_item[2],
                                ):
                                    label[i] = ner_item[0]
                                    continue
                            start += len(token) + 1
                            sentence.append(token)

                        sentences.append(" ".join(sentence).strip())
                        labels.append(label)

        return sentences, labels, ners
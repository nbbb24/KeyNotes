# 🎤 Speech and audio evaluation: choose the unit of error

Speech recognition, spoken-language understanding, synthesis, retrieval, and event detection solve different problems. Their metrics should not be substituted for one another simply because all inputs involve audio.

## 🧭 Transcription: WER, CER, and related rates

$$
WER=\frac{S+D+I}{N},
$$

where substitutions, deletions, and insertions come from an edit alignment, and N is the number of reference words. CER uses characters; phoneme/token error rates use their stated units. Define normalization, punctuation, case, segmentation, and how empty references are handled. WER can exceed 100% when insertions are numerous.

Reference `the cat sat` and hypothesis `the cat sits here` have one substitution and one insertion, giving WER $2/3$. A word error on a critical name or negation may matter more than several harmless formatting differences, so add task-specific checks.

## 🔍 Spoken-language understanding

Intent accuracy measures the selected intent. Slot F1 measures extracted entities/values; specify whether matches require exact spans and labels. Token-level and entity-level F1 differ. A semantic error rate may combine intent and slot errors, but denominators and insertion/substitution rules vary by benchmark. State the exact definition rather than treating one formula as universal.

## 📌 Speech synthesis

MOS collects human ratings under a defined protocol, often on a five-point scale. Comparative MOS uses paired relative judgments. Rater instructions, listening conditions, sample selection, and confidence intervals affect interpretation. Naturalness, intelligibility, speaker similarity, and preference are separate dimensions.

Mel-cepstral distortion, F0 RMSE/correlation, and duration error provide objective diagnostics but require alignment and voiced/unvoiced conventions. They do not replace every aspect of human listening evaluation.

## 🔎 Retrieval and sound events

Recall@K for a single known paired match is often the fraction of queries whose match appears in top K. With multiple relevant items, distinguish Hit@K from document-level recall. Cosine similarity is a score used for ranking, not by itself an evaluation of retrieval correctness. Mean average precision summarizes a relevance ranking under a stated relevance set.

Event-based F1 requires a matching label and time boundaries within defined tolerances, often onset/offset collars. Segment-based metrics evaluate activity in fixed windows. In a common segment error-rate decomposition, substitutions are $\min(FN,FP)$, deletions $\max(0,FN-FP)$, and insertions $\max(0,FP-FN)$; summing raw FN+FP+substitutions would double-count errors.

## 📊 Report enough to reproduce the score

Include dataset/split, units, preprocessing, matching tolerances, micro/macro averaging, and uncertainty. A score without these conventions is difficult to compare.

**Exercise.** Can a system have low WER but poor intent accuracy? Yes. A small number of semantically important errors can change the user's intent.

References: [Whisper evaluation](https://arxiv.org/abs/2212.04356), [sed_eval event and segment definitions](https://tut-arg.github.io/sed_eval/sound_event.html).

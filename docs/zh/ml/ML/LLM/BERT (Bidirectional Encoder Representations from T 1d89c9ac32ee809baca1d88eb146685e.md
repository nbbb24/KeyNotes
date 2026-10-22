# BERT (Bidirectional Encoder Representations from Transformer)

e pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications.

# Pre-train

## Input

[cls] at the start

[sep] segment

![image.png](BERT%20(Bidirectional%20Encoder%20Representations%20from%20T/image.png)

token + segment + position embedding

## Masked LM

masked/ replaced by other words/ keep

## Next Sentence Prediction

![image.png](BERT%20(Bidirectional%20Encoder%20Representations%20from%20T/image%201.png)

Turn into classification

50% of the time they are consecutive sentences

50% of the time they are randomly paired

# Fine tuning

## classification

[cls]→ MLP→output

## Q&A

predict probability of start and end of the input

## Named Entity Recognition
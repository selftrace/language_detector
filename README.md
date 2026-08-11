The dataset is from Kaggle (Language Detection Dataset).

Basically it gives the model a bunch of text examples where each sentence already has the correct language label. The model then tries to learn the patterns that make each language different.

The dataset has two main columns:

- `Text` → the actual sentence
- `Language` → what language the sentence belongs to

I shuffled the data, cleaned the text a bit, split it into train/test sets and used it to compare different ML models.

The dataset is from kaggle (language detection dataset).

Basically it gives the model a bunch of text examples where each sentence already has the correct language label. The model then tries to learn the patterns that make each language different.

The dataset has two main columns:

- `Text` → the actual sentence
- `Language` → what language the sentence belongs to

I shuffled the data, cleaned the text a bit, split it into train/test sets and used it to compare different ML models.


updated, this version now follows:

character-level NLP → TF-IDF → scikit-learn Pipelines → Linear SVM → stratified sampling → 5-fold CV → GridSearchCV → hyperparameter tuning → macro F1 → multiclass error analysis → robustness by text length → model persistence → inference

**FAKE NEDWS DETECTION**

This project fine-tunes a pre-trained RoBERTa model to classify fake vs. true news articles using Hugging Face's `transformers` library and PyTorch. The dataset used is derived from `Fake.csv` and `True.csv`.

## ROBERTA
This flavor was created because authors believed that BERT is hugely under-trained. There was not enough data to train BERT, 10 times more training was applied (16GB vs. 160GB). Model is bigger with 15% more parameters. Next sentence prediction is removed from BERT because the authors claimed there is no use. 4 times more masking task to learn by dynamic masking pattern.

WE recieved the feedback from our TA Ge Yao and we have decided to switch to RoBERTa instead of the conventional BERT model. This will accounts for all the capitalization found regurlary in Fake News.

## Dataset
The dataset consists of two files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains legitimate news articles
After merging:
- Data was shuffled and split into 'train.csv', 'val.csv', and 'test.csv'
- The sets are vided using a stratified approach to maintain class balance, as noted in roberta.ipynb.

## Preprocessing
- Removed nulls, duplicates
- Cleaned titles and content
- Used Hugging Face RobertaToeknizer with truncation and padding
- Converted to DatasetDict (train/val/test)

## Training Configuration
- 
TrainingArguments(
    output_dir="./results",
    // should always use "epoch" for best results
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    //it is most common to take around 32 for batch size
    per_device_train_batch_size=32, 
    // sometimes less or more but the more small the more the change model update 
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    // for optimization, use 0.05
    weight_decay=0.01,
    logging_dir='./logs',
    // if we overfit by accident then we will load the best model through checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2
)

If you train the model on eval, the metric will be random guessing. In this case, most of the models that use RoBERTa will have 50% chance of being right on each guess because 
Fake news (label = 0)
Real news (label = 1)
Then accuracy = 0.5 X 1 + 0.5 X 0 = 0.5
If dataset is imbalanced then it would always guess the majority class (in our case it would be Fake).

We could have tried custom weighting in our model such as this function, but significant loading time and our computational platform (aka computers) do not have the ability to do that.

We have done BERT uncased in the past to test our data, however, the results are not significant because Fake News often employ capitalization to emphasize. This is why in our final model we attempted to switch to RoBERTa cased, a model that accounts for that.

## EVALUATION METRICS
Accuracy, Precision, Recall, F1 Score

## FINE TUNING 
Fine-Tuning with Hugging Face's Trainer API
To simplify and streamline the training loop, we use Hugging Face's Trainer API. The Trainer abstracts away the boilerplate needed for:
- Computing loss
- Backpropagating gradients
- Optimizing model weights
- Periodically evaluating performance

I am using the method of Full Fine-Tuning this PreTrained Model. 
1. Add any additional layers on top while updating the entire whole model on labeled data. 
I loaded the roberta-base model using: model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
No layers were freezed, there were embeddings, encoders, and classification head that are all trainable BY DEFAULT. All TrainingArguments and Trainer are used without any layer freezing.
2. All aspect of the model will be updated. 
This is usually the slowest but has the highest performance.# roberta-fake-news-detection

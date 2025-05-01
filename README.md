## FAKE NEWS DETECTION

This project fine-tunes a pre-trained RoBERTa model to classify fake vs. true news articles using Hugging Face's `transformers` library and PyTorch. The dataset used is derived from `Fake.csv` and `True.csv`.

## ROBERTA
This flavor was created because authors believed that BERT is hugely under-trained. There was not enough data to train BERT, 10 times more training was applied (16GB vs. 160GB). Model is bigger with 15% more parameters. Next sentence prediction is removed from BERT because the authors claimed there is no use. 4 times more masking task to learn by dynamic masking pattern.
We use HuggingFace's Trainer API tokens. To leverage the full functionality of the Hugging Face ecosystem — including downloading pre-trained models like roberta-base and optionally pushing fine-tuned models to the Hugging Face Hub — we authenticate using a Hugging Face access token. After logging in, the token allows us to:
1. Pull pre-trained transformer models via from_pretrained()
2. Push our trained models and checkpoints to the Hugging Face Hub (if push_to_hub=True in TrainingArguments)
3. Use tokenizers directly from the 🤗 Transformers library
   
![ChatGPT Image May 1, 2025, 05_34_54 AM](https://github.com/user-attachments/assets/6ece604d-4bb0-4c52-aaf0-6e25faf01b03)

WE recieved the feedback from our TA Ge Yao and we have decided to switch to RoBERTa instead of the conventional BERT model. This will accounts for all the capitalization found regurlary in Fake News.

## Dataset
The dataset consists of two files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains legitimate news articles
After merging:
- Data was shuffled and split into 'train.csv', 'val.csv', and 'test.csv'
- The sets are vided using a stratified approach to maintain class balance, as noted in roberta.ipynb.

![test_label_distribution](https://github.com/user-attachments/assets/252d56b6-4898-4489-824a-6ed61b9e6224)
![train_label_distribution](https://github.com/user-attachments/assets/64fce49c-6c8e-4a71-9ea5-4113dc532010)
![val_label_distribution (1)](https://github.com/user-attachments/assets/b8054670-9ce3-4c2f-92d7-cffefa0664fd)

## Preprocessing
- Removed nulls, duplicates
- Cleaned titles and content
- Used Hugging Face RobertaToeknizer with truncation and padding
- Converted to DatasetDict (train/val/test)

## Training Configuration
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

<img width="691" alt="Screen Shot 2025-05-01 at 4 56 51 AM" src="https://github.com/user-attachments/assets/acfb6334-d316-4e69-b922-01a08ea77142" />

<img width="688" alt="Screen Shot 2025-04-30 at 11 01 54 PM" src="https://github.com/user-attachments/assets/fb088c13-aa84-4393-bca4-5a66c4d8b6d8" />

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

![ChatGPT Image May 1, 2025, 05_29_13 AM](https://github.com/user-attachments/assets/eeaed7d7-110a-4977-b4bd-05aad4771810)

## AREAS FOR IMPROVEMENT
***Custom Weights***
<img width="1103" alt="Screen Shot 2025-04-30 at 11 21 39 PM" src="https://github.com/user-attachments/assets/dc5a980f-e842-4567-be40-89ae00b65ec1" />

One key area of improvement introduced in this code is the implementation of **class-weighted loss** through a customized `WeightedTrainer`. By computing class weights using `sklearn.utils.class_weight` and passing them into `torch.nn.CrossEntropyLoss`, the model compensates for potential class imbalances during training. This adjustment ensures that the model doesn’t disproportionately favor the majority class, thus improving performance metrics like **F1 score**, **recall**, and **precision**—especially on underrepresented classes. The use of a custom `compute_loss` method within `WeightedTrainer` allows this weighted loss function to be integrated seamlessly into Hugging Face's `Trainer` API. Additionally, enabling `push_to_hub=True` promotes reproducibility and sharing, making this setup both robust and collaborative. However, further improvements could involve experimenting with **dynamic loss weighting**, **focal loss**, or **oversampling techniques** to further enhance model generalization on highly skewed datasets.

***Training Time Efficiency***
<img width="499" alt="Screen Shot 2025-05-01 at 6 08 54 AM" src="https://github.com/user-attachments/assets/fcc4397c-20c6-48f3-8f8c-4c6cdee21305" />

One significant area for improvement in our current pipeline is the training time efficiency. Fine-tuning roberta-base on the full dataset across multiple epochs resulted in training sessions exceeding 6 hours, which, while typical for large transformer models, can limit experimentation and iterative development. To accelerate future runs without sacrificing too much performance, I switched to a lighter model such as distilroberta-base, which is approximately 2× faster.
To improve this, we HAD to make these changes:
Reducing the number of epochs for preliminary tests.
Increasing the batch size (if GPU memory allows).
Enabling mixed precision training (fp16) to speed up computation.
Using smaller data subsets during prototyping stages.

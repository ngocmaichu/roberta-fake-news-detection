## FAKE NEWS DETECTION

This project fine-tunes a pre-trained RoBERTa model to classify fake vs. true news articles using Hugging Face's `transformers` library and PyTorch. The dataset used is derived from `Fake.csv` and `True.csv`.

## ROBERTA
RoBERTa was created because authors believed that BERT is hugely under-trained. There was not enough data to train BERT, 10 times more training was applied (16GB vs. 160GB). Model is bigger with 15% more parameters. Next sentence prediction is removed from BERT because the authors claimed there is no use. 4 times more masking task to learn by dynamic masking pattern.

In my project, I used HuggingFace's Trainer API tokens. To leverage the full functionality of the Hugging Face ecosystem (including downloading pre-trained models like roberta-base and optionally pushing fine-tuned models to the Hugging Face Hub), I authenticated using a Hugging Face access token. After logging in, the token allows us to:
1. Pull pre-trained transformer models via from_pretrained()
2. Push our trained models and checkpoints to the Hugging Face Hub (if push_to_hub=True in TrainingArguments)
3. Use tokenizers directly from the Hugging Face Transformers library

<img width="516" alt="Screen Shot 2025-05-01 at 8 40 04 AM" src="https://github.com/user-attachments/assets/fb571ce1-c375-4a22-8e20-5ae05e878ea7" />

![ChatGPT Image May 1, 2025, 05_34_54 AM](https://github.com/user-attachments/assets/6ece604d-4bb0-4c52-aaf0-6e25faf01b03)

Large Language Models (LLMs), built upon the Transformer architecture, are powerful AI systems trained on extensive text data to understand and generate human-like language, code, and more! Fine-tuning BERT for classification involves appending a task-specific layer to the pre-trained model and training it on labeled data. This process enables BERT to tailor its deep contextual understanding to the target task. In this notebook, we introduce the concept of LLMs with a focus on BERT and demonstrate how to fine-tune it for the task of fake news detection. I recieved the feedback from our TA Ge Yao in CS 506 and we have decided to switch to RoBERTa instead of the conventional BERT model. This will account for all the capitalization found regurlary in Fake News.

## Dataset
The dataset consists of two files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains legitimate news articles

After merging:
- Data was shuffled and split into 'train.csv', 'val.csv', and 'test.csv'
- The sets are divided using a stratified approach to maintain class balance, as noted in roberta.ipynb.

![output](https://github.com/user-attachments/assets/77a33b62-5273-4ab3-8da6-57ad5113e2bb)

## Preprocessing
- Removed nulls, duplicates
- Cleaned titles and content
- Used Hugging Face RobertaToeknizer with truncation and padding
- Converted to DatasetDict (train/val/test)

## Training Configuration
If you train the model on eval, the metric will be random guessing. In this case, most of the models that use RoBERTa will have 50% chance of being right on each guess because 
- Fake news (label = 0)
- Real news (label = 1)
Then accuracy = 0.5 X 1 + 0.5 X 0 = 0.5.
If dataset is imbalanced, then it would always guess the majority class (in our case it would be Fake).

<img width="691" alt="Screen Shot 2025-05-01 at 4 56 51 AM" src="https://github.com/user-attachments/assets/acfb6334-d316-4e69-b922-01a08ea77142" />

We could have tried custom weighting in our model such as this function, but significant loading time and our computational devices do not have the ability to do that.

We have done BERT uncased in the past to test our data, however, the results are not significant because Fake News often employ capitalization to emphasize. This is why in our final model we attempted to switch to RoBERTa cased, a model that accounts for that.

## EVALUATION METRICS
Accuracy, Precision, Recall, F1 Score
**🔎 Evaluation Metrics**
{'eval_loss': 0.692, 'eval_accuracy': 0.523, 'eval_precision': 0.0, 'eval_recall': 0.0, 'eval_f1': 0.0}
eval_loss ≈ 0.69 → Close to log(2) ≈ 0.693, which suggests the model is making random (uninformed) predictions.
eval_accuracy ≈ 52% → Slightly above random guessing (50% in a balanced binary classification), likely due to label imbalance.
eval_precision, recall, f1 = 0.0 → The model isn't predicting the positive class (1) — it's predicting class 0.
'train_loss': 0.7037 → This is due to the model being only trained for 0.15 epochs, which is not enough for meaningful learning — ~15% of one full pass through the training set will not ensure RoBERTa is not trained. Follow my instructions below to run for a better result.
'train_runtime': total ~40 minutes (CPU). Completed 1000 steps, but not a full epoch, which explains why metrics look underwhelming. Again, look at the recommendations below for a comprehensive recommendation.
**CONCLUSION**
Although this result reflects only 15% of the first epoch, the model shows early signs of learning, with losses near the expected binary baseline (train: 0.7037, eval: 0.6923) and an evaluation accuracy of 52.3%, which is slightly above random. The stable runtime and consistent throughput confirm a reliable training pipeline, providing a solid foundation for future improvements through longer training, class balancing, or tuning.

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
This is usually the slowest but has the highest performance.

## Roberta-Fake-News-Detection

![ChatGPT Image May 1, 2025, 05_29_13 AM](https://github.com/user-attachments/assets/eeaed7d7-110a-4977-b4bd-05aad4771810)

## AREAS FOR IMPROVEMENT
***Custom Weights***

<img width="1103" alt="Screen Shot 2025-04-30 at 11 21 39 PM" src="https://github.com/user-attachments/assets/dc5a980f-e842-4567-be40-89ae00b65ec1" />

One key area of improvement introduced in this code is the implementation of **class-weighted loss** through a customized `WeightedTrainer`. By computing class weights using `sklearn.utils.class_weight` and passing them into `torch.nn.CrossEntropyLoss`, the model compensates for potential class imbalances during training. This adjustment ensures that the model doesn’t disproportionately favor the majority class, thus improving performance metrics like **F1 score**, **recall**, and **precision**—especially on underrepresented classes. The use of a custom `compute_loss` method within `WeightedTrainer` allows this weighted loss function to be integrated seamlessly into Hugging Face's `Trainer` API. Additionally, enabling `push_to_hub=True` promotes reproducibility and sharing, making this setup both robust and collaborative. However, further improvements could involve experimenting with **dynamic loss weighting**, **focal loss**, or **oversampling techniques** to further enhance model generalization on highly skewed datasets.


***Training Time Efficiency***

<img width="499" alt="Screen Shot 2025-05-01 at 6 08 54 AM" src="https://github.com/user-attachments/assets/fcc4397c-20c6-48f3-8f8c-4c6cdee21305" />

One significant area for improvement in our current pipeline is the training time efficiency. Fine-tuning roberta-base on the full dataset across multiple epochs resulted in training sessions exceeding 6 hours, which, while typical for large transformer models, can limit experimentation and iterative development. 
To improve this, we had to make these changes in the training args:
- Reducing the number of epochs for preliminary tests
- Increasing the batch size (if GPU memory allows)
- Enabling mixed precision training (fp16) to speed up computation
- Using smaller data subsets during prototyping stages
I can also freeze dataset if possible. **Training the model** for **longer** like the example I have given above would significantly improve our results, especially if we are evaluating under a random baseline and a majority class baseline. Without changing the model and reducing the size and magnitude during the training process to test, the majority baseline would be set to the majority class, which in this case index=0 or False News. 

<img width="747" alt="Screen Shot 2025-05-01 at 8 10 39 AM" src="https://github.com/user-attachments/assets/ca12a18b-1052-4a93-ab45-a3deb77f1d83" />

This is the Hugging Face model: https://huggingface.co/ngocmaichu/roberta

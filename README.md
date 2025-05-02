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
- The sets are divided using a stratified approach to maintain class balance, as noted in tuned2.ipynb.

![output](https://github.com/user-attachments/assets/402abdbe-09c6-42c3-ae9e-3b9e226e9985)

## Preprocessing
1. Text Merging: Combined relevant text fields (e.g., title and text) into a single feature column for consistent input.
2. Label Encoding: Labeled real news as 1 and fake news as 0, following standard binary classification convention.
3. Null Handling: Removed or filled any missing values to avoid interruptions during tokenization or vectorization.
4. Train/Val/Test Split: Stratified the dataset into 60% training, 20% validation, and 20% test sets, while preserving the class balance (real news ~54.8%) across all splits.
5. Class Balance Verification: Ensured proportional class distributions after splitting using label frequency checks.
6. Tokenizer Application: Hugging Face’s AutoTokenizer with truncation=True and padding=True. TfidfVectorizer on lowercased text after removing punctuation and stopwords.

## Training Configuration
training_args = TrainingArguments(
    output_dir="./results",               # model checkpoints
    logging_dir="./logs",                 # storing logs
    num_train_epochs=2,                   # passing twice through whole dataset
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=1e-4,                   # learning rate 0.0001
    load_best_model_at_end=True,          # load best model after training
    warmup_steps=100,                     # learning rate scheduler/warmup steps
    weight_decay=0.01,                    # regularization
    logging_steps=50,                     # log every 50 steps
    log_level='info',
    save_strategy="epoch",                # save after each epoch
    eval_strategy="epoch",                # evaluate after each epoch
    eval_steps=50
)

If you train or evaluate the model improperly, such as directly on the evaluation set without a proper train/val/test split, the performance metric will reflect random guessing. For binary classification with labels for Fake News (0) and Real News (1), most RoBERTa-based models will predict either class with about a 50% chance if not properly trained. This results in an expected accuracy of 0.5 (50% of fake news guessed correctly and 0% of real news), offering no meaningful learning. 

Moreover, if the dataset is imbalanced, as in our case, where 1 News dominates, the model may default to always predicting the majority class, leading to deceptively high accuracy but poor recall and precision for the minority class. Proper splitting and training are essential to avoid this pitfall.

We also fine-tuned the Roberta-base model using the Trainer API of Hugging Face. The best-performing configuration consisted of a batch size of 64, learning rate of 3e-5, and was trained for 2 epochs. Each epoch, the model was validated against the validation set, and the best checkpoint in terms of F1 score was saved. This configuration achieved high performance with stable training. We also used early stopping and a linear learning rate scheduler with warm-up steps to avoid overfitting. Compared to earlier trials, this configuration more optimally balanced precision and recall and avoided suspicious overfitting seen in highly optimized setups.

## EVALUATION METRICS

<img width="193" alt="Screen Shot 2025-05-01 at 9 57 55 PM" src="https://github.com/user-attachments/assets/83bf39c6-d308-40dc-866a-8baa31291919" />

Majority - Acc: 0.5483 | Prec: 0.5483 | Rec: 1.0000 | F1: 0.7083  
Random   - Acc: 0.4898 | Prec: 0.5384 | Rec: 0.4880 | F1: 0.5119  
Tfidf_lr - Acc: 0.9850 | Prec: 0.9827 | Rec: 0.9901 | F1: 0.9864

Majority Class → Always predicts class 1 (real news), resulting in high recall (1.0) but biased predictions.
Random Guessing → Produces metrics near chance level, confirming the need for actual learning.
TF-IDF + Logistic Regression → Achieves exceptionally strong results with:
Accuracy of 98.50% and F1 Score of 0.9864
Balanced and high precision/recall → This shows that the model is learning to distinguish both fake and real news with strong generalization.

**CONCLUSION**
This evaluation clearly demonstrates the success of classical ML approaches when combined with well-engineered features. The TF-IDF + Logistic Regression model significantly outperforms both the majority and random baselines. With nearly 99% recall and precision, the classifier is both accurate and reliable across classes. These results indicate that even without neural networks, **strong text representations** and **consistent preprocessing** can drive competitive fake news detection. The results are trustworthy, interpretable, and fast to compute, making this setup ideal for baseline deployment or further experimentation.

## FINE TUNING 
{'eval_loss': 5.54e-05,  
 'eval_accuracy': 1.0,  
 'eval_precision': 1.0,  
 'eval_recall': 1.0,  
 'eval_f1': 1.0,  
 'eval_Validation TP': 4284,  
 'eval_Validation FP': 0,  
 'eval_Validation FN': 0,  
 'eval_Validation TN': 4696}

eval_loss ≈ 0.00005 → Almost zero loss on validation, indicating the model memorized the data.
eval_accuracy = 1.0 → Perfect accuracy across the validation set.
eval_precision, recall, f1 = 1.0 → The model predicted every instance correctly, with no false positives or false negatives.
TP = 4284, FP = 0, FN = 0, TN = 4696 → Confirms zero classification error.

While these results may initially appear ideal, such perfect performance typically raises concerns about overfitting or data leakage, especially when working with complex models like RoBERTa. Given the suspiciously flawless precision, recall, and F1 scores, it is likely that the model has either seen the validation data during training or has been inadvertently trained on non-separated splits. Although the training pipeline executed successfully, these results do not reflect real-world generalization. We strongly recommend revisiting data splitting and ensuring clean separation of train/val/test sets before drawing final conclusions. 
***If resolved, RoBERTa's architecture still offers great promise for robust fake news detection in future iterations.***

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

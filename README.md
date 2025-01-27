

## Assignment
### Step 1: Implement a Sentence Transformer Model
● Implement a sentence transformer model using any deep learning framework of your
choice. This model should be able to encode input sentences into fixed-length
embeddings.

● Test your implementation with a few sample sentences and showcase the obtained
embeddings.

● Discuss any choices you had to make regarding the model architecture outside of the
transformer backbone


## Step 2: Multi-Task Learning Expansion
Expand the sentence transformer model architecture to handle a multi-task learning setting.

● Task A: Sentence Classification
○ Implement a task-specific head for classifying sentences into predefined classes
○ Classify sentences into predefined classes (you can make these up).

● Task B: [Choose an Additional NLP Task]
○ Implement a second task-specific head for a different NLP task, such as Named
Entity Recognition (NER) or Sentiment Analysis (you can make the labels up).

● Discuss the changes made to the architecture to support multi-task learning.
Note that it’s not required to actually train the multi-task learning model or implement a training
loop. The focus is on implementing a forward pass that can accept an input sentence and output
predictions for each task that you define.


## Step 3: Discussion Questions
1. Consider the scenario of training the multi-task sentence transformer that you
implemented in Task 2. Specifically, discuss how you would decide which portions of the
network to train and which parts to keep frozen.
For example,
● When would it make sense to freeze the transformer backbone and only train the
task specific layers?
● When would it make sense to freeze one head while training the other?
2. Discuss how you would decide when to implement a multi-task model like the one in this
assignment and when it would make more sense to use two completely separate models
for each task.
3. When training the multi-task model, assume that Task A has abundant data, while Task
B has limited data. Explain how you would handle this imbalance.
## Requirements

Stated here for clarity:
- **Simplicity**: The purpose of this project is to demonstrate knowledge rather than train a state-of-the-art model. 
- **Reproducibility**: Any user should be able to easily run the script and to obtain the same results as the author

## Setup

1. Ensure `Homebrew` is installed.  If not, go to https://brew.sh/
2. Ensure Python 3.11 is installed.  If not, please get the interpreter using `brew install python@3.11` if using MacOS
3. Ensure Make is installed. If not, go to https://formulae.brew.sh/formula/make
4. At root, type `make setup` to install virtual environment


## Expected output

Run `python3.11 question1.py` and `python3.11 question2.py` for output.  

Step 1:
```
Executing similar sentences..
Sentences: ('this is a tangerine', 'this is an orange')
Cosine similarity of sentences: 0.67
Length of Embeddings 384, 384

Executing different sentences..
Sentences: ('A man is on a tree', 'a dog is in the car')
Cosine similarity of sentences: -0.08
Length of Embeddings 384, 384

Cosine similarity check passed!
```


Step 2
```
Output - Multi-categorical Classification: [0.126, 0.132, 0.291, 0.24, 0.21]
Output - Sentiment Analysis: 0.53
```

## Question and Answer

#### Step 1: Discuss any choices you had to make regarding the model architecture outside of the transformer backbone
For the sake of quickly accomplishing the task, I had used [paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2), 
which according to the [benchmark](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) has an extremely low memory usage and the fastest
inference time.  

#### Step 2: Discuss the changes made to the architecture to support multi-task learning. 

I had converted the embedding model over to a [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased).  The outputs of this base model
is then attached to a variable list of task heads provided via dependency injection, which would generate a list of outputs with the same lengths.  

#### Step 3

**Consider the scenario of training the multi-task sentence transformer that you
implemented in Task 2. Specifically, discuss how you would decide which portions of the
network to train and which parts to keep frozen.
For example,
● When would it make sense to freeze the transformer backbone and only train the
task specific layers?
● When would it make sense to freeze one head while training the other?**


We had chosen a foundation model to avoid training from scratch.  However, we may suffer a disadvantage if the data for our use case differs
drastically from the training corpus of the model.  It's best practice to fine-tune the task-specific heads first, as these heads have not been 
properly trained on our data.  We would then evaluate the improvement in performance associated with the training as the loss function converges.   

The transformer backbone should be trained under the below cases:
1. Sufficient resources/time are approved to fine-tune a larger model
2. The data associated with the use case is drastically different from the training data of the transformer model, in which case we would determine via evaluation of the model 
performance upon the use case vs a general one.  
3. The training data can be sufficiently generalized to all of the tasks given to the multi-task model.  To verify, we would have to examine the losses associated with each
task head to ensure both all losses are decreasing during the training.  

The transformer backbone should NOT be trained when:
1. Insufficiently-sized dataset to match the size of the model.  A small dataset would lead to overfiting and decreased validation performance.  We will need to
the loss curves to ensure no divergence of the train/test losses.  
2. The cost-benefit analysis does not match: The revenue gained by fine-tuning can offset the training cost (including human labor)


**Discuss how you would decide when to implement a multi-task model like the one in this
assignment and when it would make more sense to use two completely separate models
for each task.**

A multi-task model can greatly reduce the maintenance complexity and cost by avoiding the management of multiple models.  However, as mentioned earlier, the question is whether
joint training can improve the evaluation metric of both to a satisfactory level for the project.  If not, individual training can be explored as a way of developing more specialized
models.

**When training the multi-task model, assume that Task A has abundant data, while Task
B has limited data. Explain how you would handle this imbalance.**

There are a few ways:

1. Address the imbalance at the weights level:  Appropriately weigh the contributions to the loss functions by both tasks during training. However this is assuming no corruption of Task A by the compensation of Task B.  
2. If Task B has limited data to the extend which it is not useful, not a sizable proportion, or non-generalizable, then we can discard and only use Task A to determine whether its data can be generalized to Task B.  
3. It may be possible to create a joint head connecting the two task heads, perform training on Task A, and then fine-tune Task B's head.  This is given the possibility that Task A's data can contribute beyond the Transformer backbone.  
4. Look for foundational models that is trained upon data similar to Task B, if available.  
5. There are data augmentation approaches such as generating synthetic data, but this is to be used with caution, as this may generate cases far removed from real data, especially in the NLP domain due to high dimensionality.
6. Fall back to simpler models for Task B to avoid overfitting, as part of the task head.  I'm looking at you: Naive-Bayes

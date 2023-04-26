# liteGPT
Exploration and tweaking of Karpathy's nanoGPT (from his video: "Let's build GPT")

---
## Code Changes
Changes made from the original video version of the nanoGPT code include:
* Adding CUDA support
* Added mixed precision support (Autocast with GradScalar) - about 45% speed up from initial CUDA support.
* Added CUDA auto tuning, which improved performance ~10% on the development laptop.
* Ability to adjust hyper parameters to mimic other GPT models (GPT-3 sizes do not run on the development laptop, which only has 8 GB GPU)
* Support for phased analysis of various hyper parameters

---
## Experimentation Phases
To explore hyper parameters impact on the GPT framework, four phases were run against a novella (which has 112,191 letters in length):
1) 1316 hyper parameter permutations for 500 iterations of training time.
2) From the 50 best experiments, the hyper parameters were narrowed down to 320 permutations running for 1,500 iterations of training.
3) From the best of those experiments, 45 permutations running for 5,000 iterations.
4) 8 permutations of the best hyper parameter settings for 10,000 iterations.

## Training Loss
### By Learning Rate
Higher learning rates got to a lower error rate quicker.
![Learning Rate comparision](charts/training_loss_by_lr.png)
### By Batch Size
Higher batch sizes got to a lower error rate quicker.
![Batch Size comparison](charts/training_loss_by_batch_size.png)
### By Block Size
Higher block sizes got to a lower error rate quicker.
![Block Size comparison](charts/training_loss_by_block_size.png)
### By Embeddings
The two highest embeddings counts got to lower error rates quicker, with 512 performing slightly better than 768.  This could be a function of GPU constraints.
![Embeddings Comparison](charts/training_loss_by_embeddings.png)
### By Heads
Based on the constraints (the number of heads is 1/64th of the embedding size) the analysis matches Embeddings.
![Heads Comparison](charts/training_loss_by_heads.png)
### By Layers
6 Layers were slightly better than 4 layers.
![Layers Comparision](charts/training_loss_by_layers.png)

## Hyper Parameter Feature Importance for Training Loss
Learning Rates and Embeddings/Heads are most important for quality.
![Hyper Parameter Feature Importance](charts/consolidated_hyper_parameter_importance.png)

## Hyper Parameter Feature Importance for Training Loss and Training Time
Learning Rates, Batch Size, and Layers have the most impact on training on time.  This suggest to that to find the best quality training time,
Learning Rate and Embeddings should be as higher, but Batch Size and Layers should be lower.
![Hyper Parameter Feature Importance](charts/consolidated_hyper_parameter_importance_including_training_time.png)


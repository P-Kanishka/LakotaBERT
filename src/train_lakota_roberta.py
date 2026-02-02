from pathlib import Path


from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

import torch

print("CUDA==", torch.cuda.is_available())

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("/home/usd.local/kanishka.parankusham/rizk_lab/shared/kanishka/model24/", max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

model.num_parameters()
# => 84 million parameters

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
   tokenizer=tokenizer,
   file_path="//home/usd.local/kanishka.parankusham/rizk_lab/shared/kanishka/model24/train/train.txt",
   block_size=128,
)

eval_dataset = LineByLineTextDataset(
   tokenizer=tokenizer,
   file_path="/home/usd.local/kanishka.parankusham/rizk_lab/shared/kanishka/model24/val/val.txt",
   block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

#pip install accelerate -U



from transformers import Trainer, TrainingArguments
#    per_gpu_train_batch_size=64,  to 128
training_args = TrainingArguments(
    output_dir="/home/usd.local/kanishka.parankusham/rizk_lab/shared/kanishka/model24/",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=1_000,    
    num_train_epochs=10,
    logging_steps=1_000,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128, 
    save_steps=10_000,
    save_total_limit=10,
    logging_dir='./logs',
    #prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)




import time

start_time = time.time()
trainer.train()  # Execute the model training process
end_time = time.time()

elapsed_time = end_time - start_time

print("Training time:", elapsed_time, "seconds")  # Print the training time


trainer.save_model("/home/usd.local/kanishka.parankusham/rizk_lab/shared/kanishka/model24/")

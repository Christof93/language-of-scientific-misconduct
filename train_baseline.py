import comet_ml
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from evaluate import load
import torch
import pandas
import os
os.environ["COMET_API_KEY"] = "7DD2YbCm8VVKihb0pzoErza1b"
os.environ["COMET_LOG_ASSET"] = "True"

from transformers import set_seed

metric = load("f1")

# CUDA_VISIBLE_DEVICES=1 SEED=1 python3 train_baseline.py
# CUDA_VISIBLE_DEVICES=2 SEED=2 python3 train_baseline.py
# CUDA_VISIBLE_DEVICES=3 SEED=3 python3 train_baseline.py
# CUDA_VISIBLE_DEVICES=4 SEED=4 python3 train_baseline.py
# CUDA_VISIBLE_DEVICES=5 SEED=5 python3 train_baseline.py



for seed in [os.environ["SEED"]]:
    for domain in [
        'Social_Sciences', 'Physical_Sciences', 
        'Health_Sciences', 'Life_Sciences', 'all'
    ]:
        for section in [
                'Abstract', 
            ]:
            inference_samples = pandas.read_parquet(f"./25_01_22_target_corpus.gzip")
            inference_samples = inference_samples[inference_samples["Abstract"] == inference_samples["Abstract"]]
            inference_samples = inference_samples[["OriginalPaperDOI", "Abstract"]]
            inference_samples.columns = ["OriginalPaperDOI", "text"]
            
            inference_dataset = Dataset.from_pandas(inference_samples[["text"]])
            
            for reason in ["paper_mill", "falsification", "random_content"]:
                    
                if reason != "paper_mill" and domain != "all":
                    continue

                if reason == "paper_mill" and domain != "all":
                    train_samples = pandas.read_parquet(f"./train_{reason}_samples_{domain}.gzip")
                    dev_samples = pandas.read_parquet(f"./dev_{reason}_samples_{domain}.gzip")
                    test_samples = pandas.read_parquet(f"./test_{reason}_samples_{domain}.gzip")
                else:
                    train_samples = pandas.read_parquet(f"./train_{reason}_samples.gzip")
                    dev_samples = pandas.read_parquet(f"./dev_{reason}_samples.gzip")
                    test_samples = pandas.read_parquet(f"./test_{reason}_samples.gzip")
        
                train_dataset = Dataset.from_pandas(train_samples)
                dev_dataset = Dataset.from_pandas(dev_samples)
                test_dataset = Dataset.from_pandas(test_samples)
                
        
                for model_name in [
                    "microsoft/deberta-v3-base", "bert-base-uncased", "answerdotai/ModernBERT-base", "roberta-base",
                    "allenai/scibert_scivocab_uncased", "KISTI-AI/Scideberta-full", "medicalai/ClinicalBERT"
                ]:
            

                    if os.path.exists(f"./preds_{reason}_{section}_{model_name.replace('/', '_')}_{domain}_{seed}"):
                        print("Done")
                        continue
                        
                    exp = comet_ml.Experiment(project_name=f"retract_{reason}_{section}")
                    
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
                    
                    def tokenize_function(examples):
                        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
                    
                    train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
                    dev_tokenized_datasets = dev_dataset.map(tokenize_function, batched=True)
                    test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)
                    
                    inference_tokenized_datasets = inference_dataset.map(tokenize_function, batched=True)
                    
                    train_tokenized_datasets = train_tokenized_datasets.remove_columns(["text"])
                    dev_tokenized_datasets = dev_tokenized_datasets.remove_columns(["text"])
                    test_tokenized_datasets = test_tokenized_datasets.remove_columns(["text"])
                    inference_tokenized_datasets = inference_tokenized_datasets.remove_columns(["text"])
        
                    train_tokenized_datasets = train_tokenized_datasets.rename_column("label", "labels")
                    dev_tokenized_datasets = dev_tokenized_datasets.rename_column("label", "labels")
                    test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels")
        
                    
                    training_args = TrainingArguments(
                        output_dir="./results",
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        learning_rate=2e-5,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=16,
                        num_train_epochs=5,
                        weight_decay=0.01,
                        logging_dir="./logs",
                        logging_steps=10,
                        save_total_limit=2,
                        load_best_model_at_end=True,
                        run_name=f"{section}_{model_name}",
                        report_to="comet_ml"
                    )
                    
                    def compute_metrics(eval_pred):
                        logits, labels = eval_pred
                        predictions = torch.argmax(torch.tensor(logits), dim=-1)
                        return metric.compute(predictions=predictions, references=labels)
                    
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_tokenized_datasets,
                        eval_dataset=dev_tokenized_datasets,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                    )
                    
                    trainer.train()
                    
                    eval_results = trainer.evaluate()
                    print(f"Evaluation Results: {eval_results}")
                    test_results = trainer.predict(test_tokenized_datasets)
                    print(f"Test Results: {eval_results}")
        
                    
                    inference_preds = list(trainer.predict(inference_tokenized_datasets).predictions.argmax(axis=1))
                    print(inference_preds)
                    
                    result_samples = inference_samples.copy()
                    del result_samples["text"]
                    result_samples["model_name"] = model_name
                    result_samples["section"] = section
                    result_samples["reason"] = reason
                    result_samples["preds"] = inference_preds
                    result_samples["domain"] = domain
                    result_samples["seed"] = seed
                    result_samples.to_csv(f"preds_{reason}_{section}_{model_name.replace('/', '_')}_{domain}_{seed}")
                    
                    print("Fine-tuned model saved!")
                    #model.push_to_hub(f"tresiwalde/retract_{reason}_{section}_{model_name.replace('/', '_')}_{domain}_{seed}", private=True)
                    exp.end()
import json
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


class original_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        return text

    def collate(self, unpacked_data):
        # truncation=True is needed but does nothing!
        return unpacked_data


class LLM_dataset(Dataset):
    def __init__(self, dataset: Dataset, config: dict):
        self.save_embedding_path = Path(config["save_embedding"])
        if self.save_embedding_path.not_exists() or self.save_embedding_path.is_empty():
            print("No embeddings found. Generating embeddings.")
            self.save_embedding_path.mkdir(parents=True, exist_ok=True)
            self.create_dataset(dataset, config)
        self.id = json.load(open(Path(self.save_embedding) / "mapping_id.json"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.id.keys()[idx]

    def create_dataset(self, dataset: Dataset, config: dict):
        model = AutoModelForCausalLM.from_pretrained(
            config["embed_model_path"], device_map=config["device"]
        )  # dim 768
        tokenizer = AutoTokenizer.from_pretrained(
            config["embed_model_path"], padding_side="left"
        )

        tokenizer.pad_token = tokenizer.eos_token

        dataset: Dataset = original_dataset(dataset)
        dataloader = DataLoader(
            dataset=dataset, shuffle=False, batch_size=config["batch_size"]
        )

        ids = 0
        mapping_id = {}

        hidden_states = (self.save_embedding_path / "hidden_states").mkdir(
            parents=True, exist_ok=True
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                tokenized_batch = tokenizer(
                    batch, padding=True, return_tensors="pt"
                ).to(config["device"])
                output = model.generate(
                    **tokenized_batch,
                    do_sample=True,
                    max_new_tokens=config["max_new_tokens"],
                    top_k=50,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                batch_last_hidden_state = torch.cat(
                    [i[-1] for i in output["hidden_states"]][1:], dim=1
                )

                for text, last_hidden_state in zip(batch, batch_last_hidden_state):
                    torch.save(last_hidden_state, hidden_states / f"{ids}.pt")
                    mapping_id[text] = ids
                    ids += 1

            json.dump(
                mapping_id, open(self.save_embedding_path / "mapping_id.json", "w")
            )

    def collate_fn(self, batch):
        hidden_states = []
        for text in batch:
            hidden_states.append(
                torch.load(
                    self.save_embedding_path / "hidden_states" / f"{self.id[text]}.pt"
                )
            )
        return batch, torch.stack(hidden_states)

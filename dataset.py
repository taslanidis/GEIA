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
        testing = True
        self.save_embedding_path = Path(config["save_embedding"])
        if (
            not self.save_embedding_path.exists()
            or (
                self.save_embedding_path.is_dir()
                and not any(p.is_file() for p in self.save_embedding_path.iterdir())
            )
            or testing
        ):
            print("No embeddings found. Generating embeddings.")
            self.save_embedding_path.mkdir(parents=True, exist_ok=True)
            self.create_dataset(dataset, config)
        self.id = json.load(open(Path(self.save_embedding_path) / "mapping_id.json"))
        self.keys = list(self.id.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.keys[idx]

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
            dataset=dataset,
            shuffle=False,
            batch_size=config["batch_size"] 
            # if config["batch_size"] < 32 else 32,
        )

        ids = 0
        mapping_id = {}
        output_text = {}

        hidden_states = self.save_embedding_path / "hidden_states"

        hidden_states.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                # Prepend the prompt to each item in the batch
                batch_with_prompt = [f"""The following is a friendly conversation between a human and an AI. The response of the AI is single, thoughtful, enthusiastic and engaging sentence that builds meaningfully on what the human said, adds depth to the conversation, and invites further dialogue with the human. If the AI does not know the answer to a question, it truthfully says it does not know.
                    Your task is to just output the response of the AI not the human as well.
                    Current conversation:
                    Human: {item}
                    AI:""" for item in batch]

                # Tokenize the batch with the prompt added
                tokenized_batch = tokenizer(
                    batch_with_prompt, padding=True, return_tensors="pt"
                ).to(config["device"])

                output = model.generate(
                    **tokenized_batch,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.8,
                    # num_beams=2,
                    length_penalty=1,
                    repetition_penalty=1.2,
                    max_new_tokens=config["max_new_tokens"],
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                batch_last_hidden_state = torch.cat(
                    [i[-1] for i in output["hidden_states"]][1:], dim=1
                )

                from concurrent.futures import ThreadPoolExecutor
                
                # Save the hidden states to disk
                def save_hidden_states(last_hidden_state, ids):
                    # Save the hidden state to disk
                    torch.save(last_hidden_state, f"{hidden_states}/{ids}.pt")
                    return True
                
                mapping_id.update({text: ids + i for i, text in enumerate(batch)})
                ids = mapping_id[batch[-1]]
                
                # Use a ThreadPoolExecutor for parallel saving
                with ThreadPoolExecutor() as executor:
                    # Submit tasks to save each hidden state, passing the ids to be used
                    futures =[executor.submit(save_hidden_states, last_hidden_state, mapping_id[text])
                            for text, last_hidden_state in zip(batch, batch_last_hidden_state)]

                    sequences_to_decode = [
                        sequence[tokenized_batch.input_ids.shape[1]:] for sequence in output["sequences"]
                    ]


                    # Save the outputs until the last hidden state is saved
                    output_text.update(
                        {
                            text: tokenizer.decode(sequence, skip_special_tokens=True)
                            for text, sequence in zip(batch, sequences_to_decode)
                        }
                    )
                        
                    # wait for all tasks to complete
                    [future.result() for future in futures]

                # Explicitly delete tensors and clear cache
                del tokenized_batch, output, batch_last_hidden_state
                torch.cuda.empty_cache()

            json.dump(
                mapping_id, open(self.save_embedding_path / "mapping_id.json", "w")
            )

            json.dump(
                output_text, open(self.save_embedding_path / "output_text.json", "w")
            )

<<<<<<< HEAD
        del model, tokenizer
        torch.cuda.empty_cache()
        
=======
>>>>>>> refs/remotes/origin/LLM-addition
    def collate(self, batch):
        hidden_states = []
        for text in batch:
            hidden_states.append(
                torch.load(
                    self.save_embedding_path / "hidden_states" / f"{self.id[text]}.pt"
                )
            )
        return batch, torch.stack(hidden_states)

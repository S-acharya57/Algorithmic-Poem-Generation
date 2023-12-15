def here():
    import warnings
    import random
    import os

    import time
    import datetime
    import torch

    from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup
    from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler

    warnings.filterwarnings('ignore', message= 'Series.__getitem__')
    warnings.filterwarnings('ignore', category = DeprecationWarning)


    import numpy as np
    import pandas as pd
    no_deprecation_warning=True

    import os

    # Get the directory path where this script resides
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Construct the absolute path to your CSV file
    csv_file_path = os.path.join(dir_path, 'dataset_topics_60k.csv')
    model_file_path = os.path.join(dir_path, 'gpt2_model_topicwise.pth')
    df = pd.read_csv(csv_file_path)

    # Function to create a string instance based on specific columns in a DataFrame
    def combine_columns_to_string(index):
        # Ensuring the index is non-negative
        assert index >= 0, 'Index cannot be a negative integer'

        # Retrieving the row at the specified index
        selected_row = df.iloc[index, :]

        # Combining columns 0, 1, and 2 into a single string
        # making the poem seem together at similar manner
        combined_string = str(selected_row[21])+' = '+str(selected_row[1]) +'. / ' +str(selected_row[2]) + '. / '+str(selected_row[3])

        return combined_string

    # Applying the function to each row in the df
    document = [combine_columns_to_string(i) for i in range(len(df.iloc[:, 2]))]

    #cleaning the document
    document = [string.replace('\'', '') for string in document]

    #using pretrained gpt-2 model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    special_tokens_dict = {
        'bos_token': '<BOS>',
        'eos_token': '<EOS>',
        'pad_token': '<PAD>'}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length= max([tokenizer(d, return_tensors="pt").input_ids.cuda().shape[1] for d in document])
    print(f'Max length is {max_length}')

    class PoemDataset(Dataset):
        def __init__(self, data, tokenizer, gpt2_type='gpt2', max_length=max_length):
            self.tokenizer = tokenizer
            self.input_ids = []
            self.attn_masks = []
            all_texts = ""
            # Iterate over data, tokenize each sequence and append its input_id and attention_mask to respective lists
            for i in data:
                all_texts+=i
                encodings_dict = tokenizer(all_texts,
                                        truncation=True,
                                        max_length=max_length,
                                        padding='max_length')

                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                all_texts = ""

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx]

    poem_dataset = PoemDataset(document, tokenizer, max_length=max_length)

    warning_message = "resizing the embedding layer without providing a `pad_to_multiple_of` parameter"
    warnings.filterwarnings("ignore", message=warning_message)


    #configuring the model
    configuration = GPT2Config(vocab_size=len(tokenizer), n_positions=max_length).from_pretrained('gpt2', output_hidden_states=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)
    model.resize_token_embeddings(len(tokenizer))

    split = 0.8
    train_size = int(split * len(poem_dataset))

    train_dataset, val_dataset = random_split(poem_dataset, [train_size, len(poem_dataset) - train_size])

    RAND_SEED = 73
    BATCH_SIZE = 2
    EPOCHS = 1

    poem_train_dataloader = DataLoader(train_dataset,
                                sampler=RandomSampler(train_dataset),
                                batch_size=BATCH_SIZE)

    poem_val_dataloader = DataLoader(val_dataset,
                                sampler=SequentialSampler(val_dataset),
                                batch_size=BATCH_SIZE)

    # helper function for logging time
    def format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    # hyperparameters
    learning_rate = 1e-3
    eps = 1e-8
    warmup_steps = 50
    device = torch.device('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_steps = len(poem_train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)


    # Move the model to the specific device (GPU/CPU).
    model = model.to(device)

    #loading the trained model with the poem dataset
    model.load_state_dict(torch.load(model_file_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    output_poems = []
    # create text generation seed promp
    prompts = ["<BOS> Blue sky","<BOS> Life and sadness","<BOS> Wild water "]
    for prompt in prompts:
        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(device)

        sample_outputs = model.generate(
                                        generated,
                                        do_sample=True,
                                        pad_token_id=tokenizer.pad_token_id,
                                        top_k=50,
                                        max_length=max_length,
                                        top_p=0.95,
                                        num_return_sequences=3
                                        )
        
        # print(tokenizer.decode(sample_outputs[0], skip_special_tokens=True))

        for i, sample_output in enumerate(sample_outputs):
            # print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            output_poems.append(tokenizer.decode(sample_output, skip_special_tokens=True))
    return output_poems
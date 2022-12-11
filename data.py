import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import re
from underthesea import word_tokenize

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
    "]+", flags=re.UNICODE)
def remove_emoji(text):
    return emoji_pattern.sub(r' ', text)

def remove_punctuations(text):
    return re.sub(r'[^\w\s]', '', text)

ICONS = [':)))))', ':))))', ':)))', ':))', ':)', ':D', ':v' \
    ':(((((', ':((((', ':(((', ':((', ':(' \
        '(y)', ':">', ':\'(', ':|', ':-)' 
]
def remove_icons(text):
    for icon in ICONS:
        text = text.replace(icon, ' ')
    return text


def preprocess(text):
    if text == '':
        return ''
    text = text.lower().strip().replace('\n', ' ')
    text = remove_emoji(text)
    text = remove_punctuations(text)
    text = remove_icons(text)
    return text

class SentimentDataset(Dataset):
    def __init__(self, df, max_len=256, train=True):
        self.df = df
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True, force_download=True)
        self.train = train
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        To customize dataset, inherit from Dataset class and implement
        __len__ & __getitem__
        __getitem__ should return 
            data:
                input_ids
                attention_masks
                text
                targets
        """
        row = self.df.iloc[index]
        if self.train:
            revid, text, label = self.get_input_data(row, self.train)
        else:
            revid, text = self.get_input_data(row, self.train)

        # Encode_plus will:
        # (1) split text into token
        # (2) Add the '[CLS]' and '[SEP]' token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map token to their IDS
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        
        if self.train:
            return {
                'revid': revid,
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_masks': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(label, dtype=torch.float),
            }
        else:
            return {
                'revid': revid,
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_masks': encoding['attention_mask'].flatten(),
            }

    def get_input_data(self, row, train=True):
        # Preprocessing: {remove icon, special character, lower}
        text = row['Comment']
        text = preprocess(text)
        text = ' '.join(word_tokenize(text))
        revid = row['RevId']
        if train:   
            label = row['Rating'] / 10.0
            return revid, text, label
        return revid, text

def get_dataloader(df, max_len=256, train=True, batch_size=32, num_workers=4, shuffle=True):
    ds = SentimentDataset(df, max_len, train)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
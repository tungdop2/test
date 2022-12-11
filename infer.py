import pandas as pd
import torch

from data import get_dataloader
from model import SentimentClassifier

import argparse
from tqdm import tqdm

def main(df_path, ckpt_path, output_path, device='cuda'):

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA is not available. Using CPU instead.')

    df = pd.read_csv(df_path)
    df = df.fillna('')

    # get dataloader
    dataloader = get_dataloader(df, train=False, batch_size=32, shuffle=False)

    # load model
    model = SentimentClassifier()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    # inference
    all_revid = []
    all_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            revid = batch['revid']
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_masks'].to(device)

            outputs = model(input_ids, attention_masks)
            outputs = torch.sigmoid(outputs).cpu().numpy().flatten().tolist()
            outputs = [int(x > 0.6) for x in outputs]

            all_revid.extend(revid)
            all_pred.extend(outputs)

    # save to csv
    df = pd.DataFrame({'RevId': all_revid, 'Rating': all_pred})
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, default='data/test.csv')
    parser.add_argument('--ckpt_path', type=str, default='best.pth')
    parser.add_argument('--output_path', type=str, default='output.csv')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(args.df_path, args.ckpt_path, args.output_path, args.device)
    

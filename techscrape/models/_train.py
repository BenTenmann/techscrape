import torch
from techscrape.models import *
from techscrape.utils import get_roc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    train = 0

    file_path = '../../clean_data.csv'
    data_set = LSCompanies(file_path)
    if not train:
        data_set.load_vocab('vocab.json')
    train_set, test_set = data_set.train_test_split()

    model = LSTMClassifier(embedding_dim=80,
                           hidden_dim=200,
                           vocab_size=data_set.vocab_size,
                           label_size=1,
                           learning_rate=0.001,
                           batch_size=1)

    if train:

        model, error = train_loop(model, train_set)
        torch.save(model.state_dict(), 'model.pt')
        data_set.save_vocab('vocab.json')

        sns.set_theme()
        fig, ax = plt.subplots(figsize=(9, 6))

        sns.lineplot(data=pd.DataFrame({
            'error': error,
            'iteration': list(range(len(error)))
        }),
            x='iteration', y='error')
        plt.show()
    print(
        'loading model'
    )
    model.load_state_dict(torch.load('model.pt'))

    print(
        '\nTesting...'
    )
    predictions, targets = test_loop(model, test_set)
    cls = {'tp': [],
           'fp': []}
    for threshold in np.arange(0, 1, 0.01):
        tp, fp = get_roc(predictions, targets, threshold)
        cls['tp'].append(tp)
        cls['fp'].append(fp)

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(data=pd.DataFrame(cls),
                 x='fp', y='tp',
                 ax=ax)

    plt.show()

    pd.DataFrame(cls).to_csv('roc.csv')

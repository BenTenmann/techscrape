import torch
from torch import nn, autograd
from ..utils import TerminalColors


def train_loop(model, dataset):
    size = len(dataset)
    error = []
    for batch, (X, y) in enumerate(dataset):
        # for-prop
        pred = model(X)
        loss = model.loss(pred, y)

        # back-prop
        model.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        model.optimizer.step()
        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(
                f'{TerminalColors.WARNING}prediction: {pred}: target: {y}: loss: {loss:.4f}:'
                f' :pct done: {batch / size:.3f}{TerminalColors.ENDC}'
            )
        error.append(loss)
    return model, error


def test_loop(model, dataset):
    size = len(dataset)
    predictions = []
    targets = []
    for batch, (X, y) in enumerate(dataset):
        predictions.append(model(X).item())
        targets.append(y.item())
        print(
            f'{TerminalColors.WARNING}pct done: {batch / size:.3f}{TerminalColors.ENDC}'
        )
    return predictions, targets


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, learning_rate, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.symbol_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        # self.hidden = self.init_hidden()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=learning_rate)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, company_name):
        embeds = self.symbol_embeddings(company_name)
        x = embeds.view(len(company_name), self.batch_size, -1)
        lstm_out, _ = self.lstm(x)
        y = self.hidden2label(lstm_out[-1])
        log_probs = torch.sigmoid(y)
        return log_probs

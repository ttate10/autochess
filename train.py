import os
import torch
import chess

from torch import nn
from chess import Board, pgn
from glob import glob
from torchsummary import summary
from torchvision import transforms
from collections import OrderedDict, defaultdict
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

DATA_DIR = './data'

class ChessDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.results = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}
        self.values = defaultdict(int, {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6})
        self.games = [pgn.read_game(open(file, encoding='utf-8')) for file in glob(os.path.join(data_dir, '*.pgn')) if pgn.read_game(open(file, encoding='utf-8')) is not None]

        if transform is not None:
            self.games = list(map(transform, self.games))

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        return self.encode(self.games[idx])

    def state(self, board):
        x = torch.zeros((6, 64), dtype=torch.uint8)
        for i, piece in enumerate(map(board.piece_at, range(64))):
            if piece is not None:
                symbol = piece.symbol().upper()
                x[self.values[symbol]][i] = 1 if symbol.isupper() else -1
        return x

    @staticmethod
    def encode_move(move):
        y = torch.zeros(64)
        y[chess.parse_square(move[-2:])] = 1
        return y

    def encode(self, games):
        x, y, results = zip(*[(self.state(Board.push_san(move)), self.encode_move(move), game.headers['Result']) 
                            for game in games 
                            for move in game.mainline_moves()])

        return torch.Tensor(list(x)), torch.Tensor(list(y)), torch.Tensor(list(results))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self._build_model()

    def _build_model(self):
        layers = []
        for i in range(1, 9):
            layers.append((f'conv{i}', nn.Conv2d(256 if i > 1 else 6, 256, kernel_size=3, padding=1, stride=1)))
            layers.append((f'relu{i}', nn.ReLU()))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    BATCH_SIZE = 64
    NUM_WORKERS = 6

    transform = transforms.Compose([transforms.ToTensor()])
    chess_dataset = ChessDataset(DATA_DIR, transform)
    data_loader = DataLoader(chess_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print(torch.utils.data.get_worker_info())
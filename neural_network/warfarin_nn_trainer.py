from typing import Dict, List
from Prayer import prayer
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch


"""================================== DATA =================================="""
# Data wrapper around ndarray
class WarfarinDataset(Dataset):
    """Tensor-ready wrapper for (X, y) numpy arrays."""

    def __init__(self, X:np.ndarray, y:np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


"""================================== NEURAL NETWORK =================================="""
# Feed Forward Neural Network
class FeedForwardNN(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 1) # output layer – single continuous value (mg/week)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Kaiming-uniform initialisation suited for LeakyReLU.
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.NN(x)


"""================================== TRAINER =================================="""
class trainer:
    def __init__(self, df:pd.DataFrame, target_col:str, out_dir:str, epochs:int=100, batch_size:int=64, lr:float=1e-3, seed:int=42):
        # Configure
        self.MODEL_WEIGHTS: str = 'best_nn.pt'  # where to save checkpointed weights
        self.SCALER_FILE: str = 'scaler.pkl'    # where to save fitted StandardScaler
        self.TEST_SIZE: float = 0.2

        # Parameters
        self.df = df
        self.target_col = target_col
        self.out_dir = out_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        # Variables
        self.nn_parts = {}
    
    def __items__(self) -> Dict:
        return self.nn_parts

    def save_results(self):
        # Scaler
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        with open(os.path.join(self.out_dir, f'X{self.SCALER_FILE}'), 'wb') as filo:
            pickle.dump(self.nn_parts['loader']['scaler_X'], filo)
        
        with open(os.path.join(self.out_dir, f'y{self.SCALER_FILE}'), 'wb') as filo:
            pickle.dump(self.nn_parts['loader']['scaler_y'], filo)
        
        # Model
        torch.save(self.nn_parts['Result'].state_dict(), os.path.join(self.out_dir, self.MODEL_WEIGHTS))

        print(
            f'[*] Model saved to {self.MODEL_WEIGHTS}\n'
            f'[*] Scaler saved to X{self.SCALER_FILE}/y{self.SCALER_FILE}'
        )

    def initrialize(self):
        """------------------ 1) Load data ------------------"""
        print('[*] Loading data...')
        # Split dataset into features and target
        X = df.drop(columns=[self.target_col]).values.astype(np.float32)
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(df[[self.target_col]].values.astype(np.float32)).squeeze()

        # Train:Test -> 8:2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.TEST_SIZE,
            random_state=self.seed,
        )

        """"------------------ 2) Fit scaler ------------------"""
        print('[*] Scaling features...')
        scaler_X = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        """------------------ 3) Build datasets/loaders ------------------"""
        print('[*] Wrapping train/test datasets...')
        train_ds = WarfarinDataset(X_train_scaled, y_train)
        test_ds = WarfarinDataset(X_test_scaled, y_test)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        """------------------ 4) Model/optimiser ------------------"""
        print('[*] Initializing neural network:')
        device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f'Using {device}.')

        model = FeedForwardNN(X_train.shape[1]).to(device)
        print(f'Model:\n{model}\n')

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, factor=0.5)
        # print(
        #     f'\tCirterion:\n{criterion}\n\n'
        #     f'\tOptimizer:\n{optimizer}\n\n'
        #     f'\tScheduler:\n{scheduler}\n\n'
        # )

        if self.nn_parts == {}:
            self.nn_parts = {
                'loader': {
                    'train': train_loader,
                    'test': test_loader,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y,
                },
                'model': {
                    'model': model, 
                    'criterion': criterion,
                    'optimizer': optimizer, 
                    'scheduler': scheduler,
                    'device': device,
                },
                'Score': {
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': float('inf'),
                },
                'Result': None,
            }
        else:
            # loader
            self.nn_parts['loader']['train'] = train_loader
            self.nn_parts['loader']['test'] = test_loader
            self.nn_parts['loader']['scaler_X'] = scaler_X
            self.nn_parts['loader']['scaler_y'] = scaler_y

            # model
            self.nn_parts['model']['model'] = model
            self.nn_parts['model']['criterion'] = criterion
            self.nn_parts['model']['optimizer'] = optimizer
            self.nn_parts['model']['scheduler'] = scheduler
            self.nn_parts['model']['device'] = device

    def train(self, ensemble:bool=False):
        """------------------ 1-4) Initrialize Neural Network ------------------"""
        self.initrialize()

        train_loader = self.nn_parts['loader']['train']
        test_loader = self.nn_parts['loader']['test']
        scaler_y = self.nn_parts['loader']['scaler_y']

        model = self.nn_parts['model']['model']
        criterion = self.nn_parts['model']['criterion']
        optimizer = self.nn_parts['model']['optimizer']
        scheduler = self.nn_parts['model']['scheduler']
        device = self.nn_parts['model']['device']

        """------------------ 5) Training loop ------------------"""
        print(f'[*] Start training({self.epochs} epochs):')
        best_test_rmse = float('inf')
        best_test_mae = float('inf')
        best_test_r2 = float('inf')
        best_model = None

        for epoch in range(1, self.epochs + 1):
            # ---- train ----
            model.train()
            train_losses: List[float] = []
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                y_preds = model(x_batch)
                loss = criterion(y_preds, y_batch)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # ---- test ----
            model.eval()
            test_losses: List[float] = []
            test_preds, test_targets = [], []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    
                    y_preds = model(x_batch)
                    loss = criterion(y_preds, y_batch)
                    
                    test_losses.append(loss.item())
                    test_preds.append(y_preds.cpu().numpy())
                    test_targets.append(y_batch.cpu().numpy())
            
            test_preds_np = scaler_y.inverse_transform(np.concatenate(test_preds).squeeze().reshape(-1, 1)).squeeze()
            test_targets_np = scaler_y.inverse_transform(np.concatenate(test_targets).squeeze().reshape(-1, 1)).squeeze()
            test_rmse = rmse(test_targets_np, test_preds_np)
            test_mae = mae(test_targets_np, test_preds_np)
            test_r2 = r2(test_targets_np, test_preds_np)
            
            print(f'\r\tEpoch {epoch:03d}: Train MSE = {np.mean(train_losses):.4f} | Test RMSE = {test_rmse:.4f} | Test MAE = {test_mae:.4f} | Test R² = {test_r2:.4f}        ', end='')

            # Plateau scheduler – auto LR decay if progress stalls
            scheduler.step(test_rmse)

            # Checkpoint if this epoch is best so far
            if test_mae < best_test_mae:
                best_test_rmse = test_rmse
                best_test_mae = test_mae
                best_test_r2 = test_r2

                best_model = model
                # torch.save(model.state_dict(), os.path.join(out_dir, MODEL_WEIGHTS))
            
        """------------------ 6) Done ------------------"""
        print(
            f'\n[*] Training complete, best validation MAE: {best_test_mae:.4f} | RMSE: {best_test_rmse:.4f} | R²: {best_test_r2:.4f}.\n'
        )

        if best_test_mae < self.nn_parts['Score']['mae']:
            self.nn_parts['Score']['mae'] = best_test_mae
            self.nn_parts['Score']['rmse'] = best_test_rmse
            self.nn_parts['Score']['r2'] = best_test_r2
            self.nn_parts['Result'] = best_model
        
        elif best_test_mae == self.nn_parts['Score']['mae']:
            if best_test_rmse < self.nn_parts['Score']['rmse']:
                self.nn_parts['Score']['mae'] = best_test_mae
                self.nn_parts['Score']['rmse'] = best_test_rmse
                self.nn_parts['Score']['r2'] = best_test_r2
                self.nn_parts['Result'] = best_model
        
        if not ensemble:
            self.save_results()
        else:
            print(
                f'[*] Overall best validation MAE: {self.nn_parts['Score']['mae']:.4f} | RMSE: {self.nn_parts['Score']['rmse']:.4f} | R²: {self.nn_parts['Score']['r2']:.4f}.\n'
            )

    def ensemble_train(self, seeds):
        for i, seed in enumerate(seeds, start=1):
            print(f"\n=== Ensemble Model {i} (Seed: {seed}) ===")
            self.seed = seed
            self.train(ensemble=True)

        self.save_results()


Description = \
    '''
    ====================================== Wafarin Predictor Trainer ======================================
    This is the trainer for warfarin predictor.
    Result will be save as:
        1. best_nn.pt - where to save checkpointed weights.
        2. scaler.pkl - where to save fitted StandardScaler.
    
    Usage Example:
        python warfarin_nn_trainer.py -I NN_Training_Data.csv -S , -K "Therapeutic Dose of Warfarin" -O ./check_point -E 100 -BS 64 -LR 1e-3
    '''

if __name__ == '__main__':
    prayer().toPray()

    args = argparse.ArgumentParser("""""")
    args.add_argument('-I', '--INPUT', type=str, required=True, help="csv file used for training neural network.")
    args.add_argument('-S', '--SEPARATOR', type=str, default=',', help='Delimiter for input csv file, default is ",".')
    args.add_argument('-K', '--KEY', type=str, default='Therapeutic Dose of Warfarin', help='Target column for prediction, use " to include if space existed in column name. Default is "Therapeutic Dose of Warfarin".')
    args.add_argument('-O', '--OUTPUT', type=str, required=True, help="Output directory for saving result files.")
    args.add_argument('-E', '--EPOCHS', type=int, default=100, help='Epochs for training neural network, default is 100.')
    args.add_argument('-BS', '--BATCH_SIZE', type=int, default=64, help='Batch size for each round training, default is 64.')
    args.add_argument('-LR', '--LEARNING_RATE', type=float, default=1e-3, help='Learning rate for training neural network, default is 1e-3.')
    args.add_argument('-RS', '--RANDOM_SEED', type=int, default=42, help='Random seed for dataset train/test split, default is 42.')
    args.add_argument('-RST', '--RANDOM_SEED_TEST', type=int, default=0, help='Random seed test will test multiple seeds to find out a best model, enter the number of seeds you wish to test, default is 0.')

    parser = args.parse_args()

    df = pd.read_csv(parser.INPUT, sep=parser.SEPARATOR)
    trainer = trainer(
        df=df,
        target_col=parser.KEY.strip('"'),
        out_dir=parser.OUTPUT,
        epochs=parser.EPOCHS,
        batch_size=parser.BATCH_SIZE,
        lr=parser.LEARNING_RATE,
        seed=parser.RANDOM_SEED,
    )

    if parser.RANDOM_SEED_TEST == 0:
        trainer.train()
    else:
        # Ensemble training with different seeds
        seeds = np.random.randint(
            low=0, 
            high=parser.RANDOM_SEED_TEST*100, 
            size=parser.RANDOM_SEED_TEST - 1,
        )

        if parser.RANDOM_SEED not in seeds:
            seeds = np.append(seeds, parser.RANDOM_SEED)
        else:
            seeds = np.append(seeds, np.random.randint(
                                low=0, 
                                high=parser.RANDOM_SEED_TEST*100, 
                                size=1,
                                )
            )
        
        print(f'[*] Enable ensemble training with seeds:\n{seeds}')

        trainer.ensemble_train(seeds=seeds)

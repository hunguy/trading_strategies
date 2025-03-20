import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    """Import required libraries for data pipeline"""
    import vectorbt as vbt
    import pandas as pd
    import numpy as np
    from typing import Tuple, Optional
    from datetime import datetime, timedelta
    import torch
    return Optional, Tuple, datetime, np, pd, timedelta, torch, vbt


@app.cell
def _(Optional, datetime, pd, vbt):
    """Download and save historical data from Yahoo Finance"""
    def download_stock_data(
        symbol: str,
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Download historical stock data from Yahoo Finance

        Args:
            symbol: Stock symbol (e.g., 'GS' for Goldman Sachs)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to current date)
            interval: Data interval ('1d' for daily, '1h' for hourly)

        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        data = vbt.YFData.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            missing_index='drop'
        ).get(["Open", "High", "Low", "Close", "Volume"])

        return data

    # Example usage
    data = download_stock_data("GS")
    data.to_csv("GS.csv")
    return data, download_stock_data


@app.cell
def _(Tuple, data, pd, vbt):
    """Preprocess the data and create technical indicators"""
    def preprocess_data(
        df: pd.DataFrame,
        sequence_length: int = 60
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data by:
        1. Handling missing values
        2. Creating technical indicators
        3. Normalizing the data

        Args:
            df: DataFrame with OHLCV data
            sequence_length: Length of sequences for LSTM

        Returns:
            Tuple of (processed DataFrame, normalized DataFrame)
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Calculate technical indicators
        # RSI
        df['RSI'] = vbt.RSI.run(df['Close'], window=14).rsi

        # MACD
        macd = vbt.MACD.run(df['Close'])
        df['MACD'] = macd.macd
        df['MACD_Signal'] = macd.signal
        df['MACD_Hist'] = macd.hist

        # Bollinger Bands
        bb = vbt.BBANDS.run(df['Close'], window=20)
        df['BB_Upper'] = bb.upper
        df['BB_Middle'] = bb.middle
        df['BB_Lower'] = bb.lower

        # Normalize the data
        normalized_df = df.copy()
        for column in df.columns:
            if column != 'Volume':
                normalized_df[column] = (df[column] - df[column].mean()) / df[column].std()
            else:
                normalized_df[column] = df[column] / df[column].max()

        return df, normalized_df

    # Process the data
    processed_data, normalized_data = preprocess_data(data)
    return normalized_data, preprocess_data, processed_data


@app.cell
def _(Tuple, normalized_data, np, pd):
    """Split the data into training, validation, and test sets"""
    def split_data(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training, validation, and test sets

        Args:
            df: DataFrame with normalized data
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            sequence_length: Length of sequences for LSTM

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("\n=== Data Splitting ===")
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Number of features: {len(df.columns)}")
        print(f"Features: {df.columns.tolist()}")
        
        # Calculate split indices
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)

        # Create sequences
        sequences = []
        targets = []

        # Ensure we have enough data for sequences
        for i in range(len(df) - sequence_length):
            # Get sequence of data points
            sequence = df.iloc[i:(i + sequence_length)].values
            # Get target (next value after sequence)
            target = df['Close'].iloc[i + sequence_length]
            
            # Add to lists
            sequences.append(sequence)
            targets.append(target)

        # Convert to numpy arrays
        sequences = np.array(sequences)
        targets = np.array(targets)

        print("\n=== Sequence Creation ===")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Each sequence contains {sequence_length} time steps")
        print(f"Each time step has {sequences.shape[2]} features")

        # Split the data
        X_train = sequences[:train_size]
        y_train = targets[:train_size]
        X_val = sequences[train_size:train_size + val_size]
        y_val = targets[train_size:train_size + val_size]
        X_test = sequences[train_size + val_size:]
        y_test = targets[train_size + val_size:]

        print("\n=== Split Data Shapes ===")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    # Split the data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(normalized_data)
    return X_test, X_train, X_val, split_data, y_test, y_train, y_val


@app.cell
def _(Tuple, X_test, X_train, X_val, np, torch, y_test, y_train, y_val):
    """Create PyTorch DataLoaders for training"""
    def create_dataloaders(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create PyTorch DataLoaders for training

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            batch_size: Batch size for training

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        print("\n=== Converting to PyTorch Tensors ===")
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        print(f"X_train tensor shape: {X_train.shape}")
        print(f"y_train tensor shape: {y_train.shape}")
        print(f"X_val tensor shape: {X_val.shape}")
        print(f"y_val tensor shape: {y_val.shape}")
        print(f"X_test tensor shape: {X_test.shape}")
        print(f"y_test tensor shape: {y_test.shape}")

        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        print("\n=== DataLoader Information ===")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Print shape of first batch
        for batch in train_loader:
            print(f"\nFirst batch shapes:")
            print(f"Batch X shape: {batch[0].shape}")
            print(f"Batch y shape: {batch[1].shape}")
            break

        return train_loader, val_loader, test_loader

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    return create_dataloaders, test_loader, train_loader, val_loader


@app.cell
def _(torch, Tuple):
    """Define the LSTM Generator model"""
    class LSTMModel(torch.nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            output_size: int = 12,  # Changed to match input_size
            sequence_length: int = 60
        ):
            """
            LSTM-based Generator model
            
            Args:
                input_size: Number of input features
                hidden_size: Number of hidden units in LSTM
                num_layers: Number of LSTM layers
                dropout: Dropout rate
                output_size: Number of output features (should match input_size)
                sequence_length: Length of sequences to generate
            """
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.sequence_length = sequence_length
            
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size // 2, output_size * sequence_length)
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass
            
            Args:
                x: Input tensor of shape (batch_size, sequence_length, input_size)
                
            Returns:
                Output tensor of shape (batch_size, sequence_length, input_size)
            """
            # Debug prints for first batch
            if x.size(0) == 32:  # First batch
                print("\n=== Debug: LSTM Generator ===")
                print(f"Input shape: {x.shape}")
            
            # Process through LSTM
            lstm_out, _ = self.lstm(x)
            if x.size(0) == 32:  # First batch
                print(f"LSTM output shape: {lstm_out.shape}")
            
            # Generate sequence
            out = self.fc(lstm_out[:, -1, :])
            if x.size(0) == 32:  # First batch
                print(f"FC output shape: {out.shape}")
            
            # Reshape to (batch_size, sequence_length, input_size)
            out = out.view(x.size(0), self.sequence_length, -1)
            if x.size(0) == 32:  # First batch
                print(f"Final output shape: {out.shape}")
                print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
            
            return out
    
    return LSTMModel


@app.cell
def _(torch, Tuple):
    """Define the CNN Discriminator model"""
    class CNNDiscriminator(torch.nn.Module):
        def __init__(
            self,
            input_size: int,
            sequence_length: int,
            num_filters: int = 64,
            dropout: float = 0.2
        ):
            """
            CNN-based Discriminator model
            
            Args:
                input_size: Number of input features
                sequence_length: Length of input sequences
                num_filters: Number of filters in first conv layer
                dropout: Dropout rate
            """
            super(CNNDiscriminator, self).__init__()
            
            # First layer: Conv1d expects (batch_size, channels, sequence_length)
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(num_filters),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(dropout)
            )
            
            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(num_filters * 2),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(dropout)
            )
            
            self.conv3 = torch.nn.Sequential(
                torch.nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(num_filters * 4),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(dropout)
            )
            
            # Calculate the size of flattened features
            self.flatten_size = num_filters * 4 * sequence_length
            
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.flatten_size, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(512, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid()  # Ensure output is between 0 and 1
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass
            
            Args:
                x: Input tensor of shape (batch_size, sequence_length, input_size)
                
            Returns:
                Output tensor of shape (batch_size, 1) with values between 0 and 1
            """
            # Debug prints for first batch
            if x.size(0) == 32:  # First batch
                print("\n=== Debug: CNN Discriminator Input ===")
                print(f"Input shape: {x.shape}")
                print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            
            # Check for NaN values in input
            if torch.isnan(x).any():
                print("Warning: NaN values detected in input")
                x = torch.nan_to_num(x, nan=0.0)
            
            # Reshape input for CNN (batch_size, channels, sequence_length)
            # First, transpose to (batch_size, input_size, sequence_length)
            x = x.transpose(1, 2)
            
            if x.size(0) == 32:  # First batch
                print(f"After transpose shape: {x.shape}")
                print(f"After transpose range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            
            # Apply convolutional layers
            x = self.conv1(x)
            if x.size(0) == 32:  # First batch
                print(f"After conv1 shape: {x.shape}")
                print(f"After conv1 range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            
            x = self.conv2(x)
            if x.size(0) == 32:  # First batch
                print(f"After conv2 shape: {x.shape}")
                print(f"After conv2 range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            
            x = self.conv3(x)
            if x.size(0) == 32:  # First batch
                print(f"After conv3 shape: {x.shape}")
                print(f"After conv3 range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            
            # Flatten and apply fully connected layers
            x = x.view(x.size(0), -1)
            if x.size(0) == 32:  # First batch
                print(f"After flatten shape: {x.shape}")
                print(f"After flatten range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            
            x = self.fc(x)
            if x.size(0) == 32:  # First batch
                print(f"Final output shape: {x.shape}")
                print(f"Final output range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            
            # Ensure output is between 0 and 1
            x = torch.clamp(x, 0.0, 1.0)
            
            return x
    
    return CNNDiscriminator


@app.cell
def _(LSTMModel, CNNDiscriminator, torch, X_train):
    """Initialize the models and move them to GPU if available"""
    def initialize_models(
        input_size: int = X_train.shape[2],
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = X_train.shape[1]
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Initialize and configure the Generator and Discriminator models
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (generator, discriminator)
        """
        # Initialize models
        generator = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length
        )
        
        discriminator = CNNDiscriminator(
            input_size=input_size,
            sequence_length=sequence_length,
            num_filters=64,
            dropout=dropout
        )
        
        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        
        return generator, discriminator
    
    # Initialize models
    generator, discriminator = initialize_models()
    return generator, discriminator, initialize_models


@app.cell
def _(torch, Tuple):
    """Define loss functions and optimizers"""
    def setup_training_components(
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        learning_rate: float = 0.0002,
        beta1: float = 0.5
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.nn.Module, torch.nn.Module]:
        """
        Set up optimizers and loss functions for training
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            learning_rate: Learning rate for optimizers
            beta1: Beta1 parameter for Adam optimizer
            
        Returns:
            Tuple of (g_optimizer, d_optimizer, g_criterion, d_criterion)
        """
        # Optimizers
        g_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=learning_rate,
            betas=(beta1, 0.999)
        )
        
        d_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, 0.999)
        )
        
        # Loss functions
        g_criterion = torch.nn.MSELoss()
        d_criterion = torch.nn.BCELoss()
        
        return g_optimizer, d_optimizer, g_criterion, d_criterion
    
    return setup_training_components


@app.cell
def _(torch, Tuple):
    """Define the training loop for GAN"""
    def train_gan(
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        g_criterion: torch.nn.Module,
        d_criterion: torch.nn.Module,
        num_epochs: int = 100,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_critic: int = 5,  # Number of D updates per G update
        lambda_gp: float = 10.0  # Gradient penalty coefficient
    ) -> Tuple[list, list, list, list]:
        """
        Train the GAN model
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            train_loader: Training data loader
            val_loader: Validation data loader
            g_optimizer: Generator optimizer
            d_optimizer: Discriminator optimizer
            g_criterion: Generator loss function
            d_criterion: Discriminator loss function
            num_epochs: Number of training epochs
            device: Device to train on
            n_critic: Number of D updates per G update
            lambda_gp: Gradient penalty coefficient
            
        Returns:
            Tuple of (g_losses, d_losses, val_g_losses, val_d_losses)
        """
        g_losses = []
        d_losses = []
        val_g_losses = []
        val_d_losses = []
        
        def compute_gradient_penalty(discriminator, real_samples, fake_samples):
            """Compute gradient penalty for WGAN"""
            # Create interpolated samples
            alpha = torch.rand(real_samples.size(0), 1, 1).to(device)
            interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
            
            # Get discriminator output for interpolated samples
            d_interpolates = discriminator(interpolates)
            fake = torch.ones(d_interpolates.size(), requires_grad=False).to(device)
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Reshape gradients to (batch_size, -1)
            gradients = gradients.reshape(gradients.size(0), -1)
            
            # Compute gradient penalty
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty
        
        for epoch in range(num_epochs):
            g_loss = 0
            d_loss = 0
            
            # Training loop
            for batch_idx, (real_data, _) in enumerate(train_loader):
                real_data = real_data.to(device)
                batch_size = real_data.size(0)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                # Generate fake data
                z = torch.randn(batch_size, real_data.size(1), real_data.size(2)).to(device)
                fake_data = generator(z)
                
                # Ensure input dimensions are correct for discriminator
                # real_data and fake_data should be (batch_size, sequence_length, input_size)
                d_real = discriminator(real_data)
                d_fake = discriminator(fake_data.detach())  # Detach to avoid G update
                
                # Debug prints for first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    print("\n=== Debug: Discriminator Outputs ===")
                    print(f"d_real shape: {d_real.shape}")
                    print(f"d_real min: {d_real.min().item():.4f}")
                    print(f"d_real max: {d_real.max().item():.4f}")
                    print(f"d_fake shape: {d_fake.shape}")
                    print(f"d_fake min: {d_fake.min().item():.4f}")
                    print(f"d_fake max: {d_fake.max().item():.4f}")
                
                # Create labels with proper shape and device
                real_labels = torch.ones(batch_size, 1, requires_grad=False).to(device)
                fake_labels = torch.zeros(batch_size, 1, requires_grad=False).to(device)
                
                # Compute discriminator losses
                d_real_loss = d_criterion(d_real, real_labels)
                d_fake_loss = d_criterion(d_fake, fake_labels)
                
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data)
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss + lambda_gp * gradient_penalty
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                if batch_idx % n_critic == 0:
                    g_optimizer.zero_grad()
                    
                    # Generate fake data
                    z = torch.randn(batch_size, real_data.size(1), real_data.size(2)).to(device)
                    fake_data = generator(z)
                    
                    # Generator loss
                    d_fake = discriminator(fake_data)
                    g_loss = g_criterion(d_fake, real_labels)  # Generator wants to fool discriminator
                    
                    g_loss.backward()
                    g_optimizer.step()
            
            # Validation loop
            generator.eval()
            discriminator.eval()
            
            with torch.no_grad():
                val_g_loss = 0
                val_d_loss = 0
                
                for real_data, _ in val_loader:
                    real_data = real_data.to(device)
                    batch_size = real_data.size(0)
                    
                    # Generate fake data
                    z = torch.randn(batch_size, real_data.size(1), real_data.size(2)).to(device)
                    fake_data = generator(z)
                    
                    # Compute validation losses
                    d_real = discriminator(real_data)
                    d_fake = discriminator(fake_data)
                    
                    # Create labels with proper shape and device
                    real_labels = torch.ones(batch_size, 1, requires_grad=False).to(device)
                    fake_labels = torch.zeros(batch_size, 1, requires_grad=False).to(device)
                    
                    val_d_loss += d_criterion(d_real, real_labels)
                    val_d_loss += d_criterion(d_fake, fake_labels)
                    val_g_loss += g_criterion(d_fake, real_labels)
            
            # Average validation losses
            val_g_loss /= len(val_loader)
            val_d_loss /= len(val_loader)
            
            # Store losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            val_g_losses.append(val_g_loss.item())
            val_d_losses.append(val_d_loss.item())
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'G_loss: {g_loss.item():.4f}, '
                      f'D_loss: {d_loss.item():.4f}, '
                      f'Val_G_loss: {val_g_loss.item():.4f}, '
                      f'Val_D_loss: {val_d_loss.item():.4f}')
        
        return g_losses, d_losses, val_g_losses, val_d_losses
    
    return train_gan


@app.cell
def _(torch, Tuple):
    """Define the PPO training loop"""
    class PPOTrainer:
        def __init__(
            self,
            generator: torch.nn.Module,
            discriminator: torch.nn.Module,
            learning_rate: float = 0.0003,
            gamma: float = 0.99,
            epsilon: float = 0.2,
            c1: float = 1.0,
            c2: float = 0.01,
            batch_size: int = 64,
            n_epochs: int = 4
        ):
            """
            Initialize PPO trainer
            
            Args:
                generator: Generator model
                discriminator: Discriminator model
                learning_rate: Learning rate
                gamma: Discount factor
                epsilon: PPO clipping parameter
                c1: Value function coefficient
                c2: Entropy coefficient
                batch_size: Batch size for training
                n_epochs: Number of epochs per update
            """
            self.generator = generator
            self.discriminator = discriminator
            self.gamma = gamma
            self.epsilon = epsilon
            self.c1 = c1
            self.c2 = c2
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            
            # Optimizers
            self.g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
            self.d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
            
            # Memory buffers
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
        
        def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Select action using current policy"""
            with torch.no_grad():
                action = self.generator(state)
                value = self.discriminator(state)
                log_prob = torch.distributions.Normal(action, 1.0).log_prob(action)
                
            return action, value, log_prob
        
        def update(self) -> Tuple[float, float]:
            """Update policy using PPO"""
            # Convert lists to tensors
            states = torch.stack(self.states)
            actions = torch.stack(self.actions)
            old_log_probs = torch.stack(self.log_probs)
            
            # Calculate advantages
            rewards = torch.tensor(self.rewards)
            values = torch.tensor(self.values)
            advantages = rewards - values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for _ in range(self.n_epochs):
                # Generate new action probabilities
                new_actions, new_values, new_log_probs = self.select_action(states)
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
                
                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = torch.nn.MSELoss()(new_values, rewards)
                
                # Calculate entropy loss
                entropy_loss = -0.01 * torch.mean(new_log_probs)
                
                # Total loss
                total_loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
                
                # Update generator
                self.g_optimizer.zero_grad()
                total_loss.backward()
                self.g_optimizer.step()
                
                # Update discriminator
                self.d_optimizer.zero_grad()
                value_loss.backward()
                self.d_optimizer.step()
            
            # Clear memory
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
            
            return policy_loss.item(), value_loss.item()
    
    return PPOTrainer


@app.cell
def _(train_gan, PPOTrainer, generator, discriminator, train_loader, val_loader, setup_training_components):
    """Initialize and start training"""
    # Set up training components
    g_optimizer, d_optimizer, g_criterion, d_criterion = setup_training_components(
        generator, discriminator
    )
    
    # Train GAN
    g_losses, d_losses, val_g_losses, val_d_losses = train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_criterion=g_criterion,
        d_criterion=d_criterion
    )
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(generator, discriminator)
    
    return g_losses, d_losses, val_g_losses, val_d_losses, ppo_trainer


@app.cell
def _(torch, Tuple, np, pd, vbt):
    """Phase 4: PPO Training Implementation"""
    class TradingEnvironment:
        def __init__(self, data: pd.DataFrame, sequence_length: int = 60):
            """
            Trading environment for PPO training
            
            Args:
                data: DataFrame with OHLCV data and technical indicators
                sequence_length: Length of sequences for state representation
            """
            # Create a copy of the data to avoid modifying the original
            self.data = data.copy()
            
            # Calculate technical indicators if not already present
            if 'RSI' not in self.data.columns:
                self.data['RSI'] = vbt.RSI.run(self.data['Close'], window=14).rsi
                macd = vbt.MACD.run(self.data['Close'])
                self.data['MACD'] = macd.macd
                self.data['MACD_Signal'] = macd.signal
                self.data['MACD_Hist'] = macd.hist
                bb = vbt.BBANDS.run(self.data['Close'], window=20)
                self.data['BB_Upper'] = bb.upper
                self.data['BB_Middle'] = bb.middle
                self.data['BB_Lower'] = bb.lower
            
            # Normalize the data
            for column in self.data.columns:
                if column != 'Volume':
                    self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()
                else:
                    self.data[column] = self.data[column] / self.data[column].max()
            
            self.sequence_length = sequence_length
            self.current_step = sequence_length
            self.max_steps = len(data) - sequence_length
            self.initial_balance = 10000.0
            self.balance = self.initial_balance
            self.position = 0
            self.trades = []
            
            print("\n=== Trading Environment Initialization ===")
            print(f"Number of features: {len(self.data.columns)}")
            print(f"Features: {self.data.columns.tolist()}")
            print(f"Sequence length: {sequence_length}")
            print(f"Max steps: {self.max_steps}")
            
        def reset(self) -> torch.Tensor:
            """Reset the environment"""
            self.current_step = self.sequence_length
            self.balance = self.initial_balance
            self.position = 0
            self.trades = []
            return self._get_state()
        
        def _get_state(self) -> torch.Tensor:
            """Get current state representation"""
            state_data = self.data.iloc[self.current_step-self.sequence_length:self.current_step].values
            return torch.FloatTensor(state_data)
        
        def step(self, action: float) -> Tuple[torch.Tensor, float, bool, dict]:
            """
            Execute one step in the environment
            
            Args:
                action: Trading action (-1: sell, 0: hold, 1: buy)
                
            Returns:
                Tuple of (next_state, reward, done, info)
            """
            current_price = self.data['Close'].iloc[self.current_step]
            next_price = self.data['Close'].iloc[self.current_step + 1]
            
            # Calculate price change
            price_change = (next_price - current_price) / current_price
            
            # Execute action
            reward = 0
            if action > 0 and self.position <= 0:  # Buy signal
                if self.position < 0:  # Close short position
                    reward += -price_change * abs(self.position)
                self.position = 1
            elif action < 0 and self.position >= 0:  # Sell signal
                if self.position > 0:  # Close long position
                    reward += price_change * self.position
                self.position = -1
            
            # Calculate reward based on position and price change
            reward += price_change * self.position
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'action': action,
                'position': self.position,
                'price': current_price,
                'reward': reward
            })
            
            # Update step
            self.current_step += 1
            done = self.current_step >= self.max_steps - 1
            
            # Calculate info
            info = {
                'balance': self.balance,
                'position': self.position,
                'trades': self.trades
            }
            
            return self._get_state(), reward, done, info
    
    def train_ppo(
        ppo_trainer: PPOTrainer,
        env: TradingEnvironment,
        num_episodes: int = 1000,
        max_steps: int = 1000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        batch_size: int = 64,
        n_epochs: int = 4
    ) -> Tuple[list, list]:
        """
        Train the trading strategy using PPO
        
        Args:
            ppo_trainer: PPO trainer instance
            env: Trading environment
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            c1: Value function coefficient
            c2: Entropy coefficient
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            
        Returns:
            Tuple of (episode_rewards, episode_lengths)
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Collect trajectory
            while episode_length < max_steps:
                # Select action
                action, value, log_prob = ppo_trainer.select_action(state)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action.item())
                
                # Store transition
                ppo_trainer.states.append(state)
                ppo_trainer.actions.append(action)
                ppo_trainer.rewards.append(reward)
                ppo_trainer.values.append(value)
                ppo_trainer.log_probs.append(log_prob)
                
                # Update state and episode info
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Update policy
            policy_loss, value_loss = ppo_trainer.update()
            
            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f'Episode [{episode+1}/{num_episodes}], '
                      f'Reward: {episode_reward:.2f}, '
                      f'Length: {episode_length}, '
                      f'Policy Loss: {policy_loss:.4f}, '
                      f'Value Loss: {value_loss:.4f}')
        
        return episode_rewards, episode_lengths
    
    # Initialize environment and start training
    env = TradingEnvironment(processed_data)  # Use processed_data instead of raw data
    episode_rewards, episode_lengths = train_ppo(
        ppo_trainer=ppo_trainer,
        env=env,
        num_episodes=1000,
        max_steps=1000
    )
    
    return env, episode_rewards, episode_lengths, TradingEnvironment, train_ppo


if __name__ == "__main__":
    app.run()

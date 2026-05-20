import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


class Autoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.001, epochs=50, batch_size=16):
        self.input_dim    = input_dim
        self.encoding_dim = encoding_dim
        self.lr           = learning_rate
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.X_min        = None
        self.X_max        = None

        self._build()

    def _build(self):
        h1 = 1024
        h2 = 512

        # ── Encoder ──────────────────────────────────────────
        inp    = Input(shape=(self.input_dim,))
        enc    = Dense(h1,                activation='relu')(inp)
        enc    = Dense(h2,                activation='relu')(enc)
        latent = Dense(self.encoding_dim, activation='relu')(enc)

        # ── Decoder ──────────────────────────────────────────
        dec_input = Input(shape=(self.encoding_dim,))
        dec    = Dense(h2,             activation='relu')(dec_input)
        dec    = Dense(h1,             activation='relu')(dec)
        out    = Dense(self.input_dim, activation='sigmoid')(dec)

        # ── Models ───────────────────────────────────────────
        self.encoder     = Model(inp,       latent)
        self.decoder     = Model(dec_input, out)
        self.autoencoder = Model(inp,       self.decoder(latent))

        self.autoencoder.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss='mse'
        )

    def decode(self, Z):
        recon_norm = self.decoder.predict(Z, verbose=0)
        return np.clip(recon_norm, 0, 1)

    def fit(self, X):
        self.X_min = X.min()
        self.X_max = X.max()
        X_norm = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)

        self.history = self.autoencoder.fit(
            X_norm, X_norm,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=1
        )

    def encode(self, X):
        X_norm = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        return self.encoder.predict(X_norm, verbose=0)


    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Autoencoder Training Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
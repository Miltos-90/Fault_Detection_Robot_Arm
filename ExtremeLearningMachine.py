import numpy as np

class ELM(object):
    
    def __init__(self, input_size, hidden_size):
        
        # Initialise weights and biases for the hidden layer
        self.W = np.random.uniform(low = -0.5, high = 0.5, size = (input_size, hidden_size))
        self.b = np.random.uniform(low = -0.5, high = 0.5, size = [hidden_size])

    def fit(self, X, y):
        
        # Hidden layer nodes
        H = ELM.sigmoid(np.dot(X, self.W) + self.b)
        
        # Moore-penrose pseudoinverse
        H = np.linalg.pinv(H) # Moore-penrose pseudoinverse
        
        # Output weights
        self.betas = np.dot(H, y)
        
        return
        
    def predict(self, X):
        
        # Hidden layer nodes
        H = ELM.sigmoid(np.dot(X, self.W) + self.b)
        
        return np.dot(H, self.betas)
    
    @staticmethod
    def sigmoid(x):
        
        # Prevent overflow.
        x = np.clip(x, -500, 500)
        
        # Compute activation
        return 1 / ( 1 + np.exp(-x) )


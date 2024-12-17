import pickle
import gzip
from sklearn.ensemble import RandomForestRegressor

# Load the model
with open("C:\\Users\\User\\.jupyter\\Dash31\\tuned_rf_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Compress the model
with gzip.open('compressed_rf_model.pkl.gz', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

# Verify the compression worked by loading it
with gzip.open('compressed_rf_model.pkl.gz', 'rb') as f:
    loaded_model = pickle.load(f)
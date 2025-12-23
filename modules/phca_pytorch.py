import torch
import numpy as np
import ripser
from tqdm import tqdm

class PHCA_PyTorch():

    def __init__(self, dim:list=[0], device=None, use_precomputed_distances=False):
        self.dim = dim
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_precomputed_distances = use_precomputed_distances
        print(f"Using device: {self.device}")

    def fit(self, X_train, y_train):
        """ Fit the model by computing persistent homology for each class """

        # convert training data in PyTorch tensors
        if not torch.is_tensor(X_train):
            X_train = torch.from_numpy(np.array(X_train)).float()
        
        self.X_train = X_train.to(self.device)
        self.y_train = np.array(y_train)

        # organize data by class
        self.by_class_data = by_class_data(self.X_train, self.y_train, self.device)
        self.classes = list(self.by_class_data.keys())

        # compute initial persistent diagrams
        self.PD_by_class = get_persistence_diagram(
            self.by_class_data,
            self.dim,
            self.device,
            self.use_precomputed_distances
        )

        # pre-computed initial lifespans (cached for efficiency)
        self.original_lifespans = {
            clss: get_lifespan(self.PD_by_class[clss], self.dim, self.device)
            for clss in self.classes
        }

    def predict(self, X_test, batch_size=32):
        """ Predict classes for test data """

        if not torch.is_tensor(X_test):
            X_test = torch.from_numpy(np.array(X_test)).float()

        X_test = X_test.to(self.device)
        n_test = len(X_test)
        y_pred = []

        # process in batches for memory efficiency
        for batch_start in tqdm(range(0, n_test, batch_size), desc="Predicting"):
            batch_end = min(batch_start + batch_size, n_test)
            batch = X_test[batch_start:batch_end]

            # predict batch
            batch_predictions = self._predict_batch(batch)
            y_pred.extend(batch_predictions)

        return y_pred 
    
    def _predict_batch(self, X_batch):
        """ Process a batch of test points in parallel """
        batch_size = len(X_batch)
        predictions = []

        # for each test point in batch (still sequential for persistent computation)
        for i in range(batch_size):
            x_new = X_batch[i]

            # update data with new point
            by_class_data_plus = update_data(self.by_class_data, x_new, self.device)

            # compute new persistence diagrams
            PD_by_class_plus = get_persistence_diagram(
                by_class_data_plus,
                self.dim,
                self.device,
                self.use_precomputed_distances
            )

            # select class using vectorized operations
            selected_classes = class_selector(
                self.original_lifespans,
                PD_by_class_plus,
                self.dim,
                self.device
            )
            predictions.append(selected_classes)
        return predictions
    
    def predict_single(self, x_new):
        """ Predict class for a single point (legacy interface) """
        if not torch.is_tensor(x_new):
            x_new = torch.from_numpy(np.array(x_new)).float()
        x_new = x_new.to(self.device)

        return self._predict_batch(x_new.unsqueeze(0))[0]


########### HELPER FUNCTIONS ############

def by_class_data(X, y, device):
    byclass_data = {}

    for clss in np.unique(y):
        mask = y == clss
        byclass_data[clss] = X[mask]
    
    return byclass_data

def get_persistence_diagram(data, dim:list, device, use_precomputed=False):
    dgms = {}

    for key in data.keys():
        points = data[key]

        if use_precomputed and len(points) > 0:
            # compute distance matrix on GPU (may be faster)
            dist_matrix = torch.cdist(points, points, p=2)
            dist_matrix_np = dist_matrix.cpu().numpy()

            # use precomputed distances
            dgms[key] = ripser.ripser(
                dist_matrix_np,
                maxdim=max(dim),
                distance_matrix=True
            )['dgms']
        
        else:
            # standard point cloud input
            points_np = points.cpu().numpy()
            dgms[key] = ripser.ripser(points_np, maxdim=max(dim))['dgms']
    
    return dgms


def get_lifespan(PD:list, dim:list, device):
    """ Compute total lifespan using PyTorch vectorized operations """

    total_lifespan = torch.tensor(0.0, device=device)
    for d in dim:
        if len(PD[d]) == 0:
            continue
        
        # convert to torch tensor
        dgm = torch.from_numpy(PD[d]).float().to(device)

        births, deaths = dgm[:,0], dgm[:,1]

        # remove infinite values (vectorized)
        finite_mask = torch.isfinite(deaths)
        births_finite = births[finite_mask]
        deaths_finite = deaths[finite_mask]

        # compute total lifespan (vectorized sum)
        total_lifespan += deaths_finite.sum() - births_finite.sum()

    return total_lifespan


def update_data(data:dict, x_new:torch.Tensor, device):
    """ Updates each class' point cloud with new test point """
    data_plus = {}
    x_new = x_new.unsqueeze(0)

    for clss in data.keys():
        # concatenate new point to existing points
        data_plus[clss] = torch.cat([data[clss], x_new], dim=0)

    return data_plus

def class_selector(original_lifespans:dict, new_PD:dict, dim:list, device):
    """ Select class with minimum lifespan change using vectorized operations """
    # compute all new lifespans
    new_lifespans = torch.tensor([
        get_lifespan(new_PD[clss], dim, device) for clss in original_lifespans.keys()
    ])

    orig_lifespans = torch.tensor([
        original_lifespans[clss] for clss in original_lifespans.keys()
    ])

    # vectorized difference and argmin
    differences = torch.abs(new_lifespans - orig_lifespans)
    min_idx = torch.argmin(differences).item()

    return list(original_lifespans.keys())[min_idx]
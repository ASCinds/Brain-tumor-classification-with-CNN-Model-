import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

class CreateDataset(Dataset):
    """
    Class for creating a dataset that inherits the Dataset class properties.
    
    Parameters:
    - dataframe (pd.DataFrame): The input dataframe containing file paths and class labels.
    - split_type (str): Specifies whether the dataset is for training, testing, or validation.
    - transform (callable, optional): A function/transform to be applied to the image data.
    - train_size (float, optional): Proportion of data to be used for training.
    - test_size (float, optional): Proportion of data to be used for testing.
    - valid_size (float, optional): Proportion of data to be used for validation.
    - random_state (int, optional): Seed for random state to ensure reproducibility.

    Attributes:
    - dataframe (pd.DataFrame): The input dataframe containing file paths and class labels.
    - transform (callable): A function/transform to be applied to the image data.
    - data (pd.DataFrame): The split data based on the specified split_type.
    """

    def __init__(self, dataframe, split_type, transform=None, train_size=0.75, test_size=0.15, valid_size=0.1, random_state=42):
        """
        Initialize the dataset with specified parameters.
        """
        self.dataframe = dataframe
        self.transform = transform

        # Split the data into train, test, and validation sets
        train_data, test_data = train_test_split(dataframe, test_size=(test_size + valid_size), random_state=random_state)
        test_data, valid_data = train_test_split(test_data, test_size=valid_size / (test_size + valid_size), random_state=random_state)

        if split_type == 'train':
            self.data = train_data
        elif split_type == 'test':
            self.data = test_data
        elif split_type == 'valid':
            self.data = valid_data
        else:
            raise ValueError("Invalid split_type. Choose 'train', 'test', or 'valid'.")

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets the item at the specified index.

        Parameters:
        - idx (int): Index of the item.

        Returns:
        - tuple: A tuple containing the image and its corresponding label.
        """
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        return img, label

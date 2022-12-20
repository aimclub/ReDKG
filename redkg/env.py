from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from redkg.config import Config
from redkg.utils import pickle_load


class Simulator:
    """Custom Environment that follows gym interface"""

    def __init__(self, config: Config, mode: str) -> None:
        self.rating_dict = pickle_load(f"{config.preprocess_results_dir}/{mode}_data_dict.pkl")
        self.user_ids = list(self.rating_dict.keys())
        self.num_users = len(self.user_ids)

    def __len__(self) -> int:
        return len(self.rating_dict)

    def get_user_data(self, user_idx: int) -> Tuple[int, NDArray, NDArray]:
        """Get user data by user index

        :param user_idx: (int) index of user
        :returns: (Tuple[int, NDArray, NDArray]) User ID, Array with items, Array with Rates
        """
        user_id = self.user_ids[user_idx % self.num_users]
        user_ratings = np.array(self.rating_dict[user_id])
        item_ids, rates = user_ratings[:, 0].astype(np.int), user_ratings[:, 1].astype(np.float)
        return user_id, item_ids, rates

    def step(self, user_id: int, recommended_item_id: int) -> int:
        """Executes a step in the environment by applying an action. Returns the new observation and reward.

        :param user_id: Current agent (user)
        :param recommended_item_id: Action (selected item)
        :return: reward (0 if not interaction, attribute otherwise)
        """
        user_ratings = np.array(self.rating_dict[user_id])
        item_ids, rates = user_ratings[:, 0].astype(np.int), user_ratings[:, 1].astype(np.float)
        try:
            t = np.where(item_ids == recommended_item_id)[0][0]
            return rates[t]
        except IndexError:  # User did not interacted with recommended item
            return 0

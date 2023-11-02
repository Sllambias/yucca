import torch
import numpy as np


class InfiniteRandomBatchSampler(torch.utils.data.Sampler) :
	def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = None):
		assert len(dataset) > 0
		self.dataset = dataset
		self.batch_size = batch_size
	
	def __iter__(self):
		yield np.random.choice(len(self.dataset), self.batch_size)
		

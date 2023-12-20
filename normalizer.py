import statistics
import numpy as np
from sklearn.preprocessing import StandardScaler

class BaseNormalizer(object):
	def __init__(self, data):
		self.data = data
		self._normalize()


	def _normalize(self):
		pass

	def get_normalized_value(self, vec):
		pass

	def get_data_normalized(self):
		return np.array([self.get_normalized_value(vec) for vec in self.data])

class SKNormalizer(BaseNormalizer):
	def __init__(self, data):
		super().__init__(data)

	def _normalize(self):
		self.transformer = StandardScaler().fit(self.data)

	def get_normalized_value(self, vec):
		return self.transformer.transform(vec)

	def get_data_normalized(self):
		return self.transformer.transform(self.data)



class MinMaxNormalizer(BaseNormalizer):
	def __init__(self, data):
		super().__init__(data)

	def _normalize(self):
		self.min = np.min(self.data)
		self.max = np.max(self.data)

	def get_normalized_value(self, vec):
		return (vec - self.min) / (self.max - self.min)


class ZNormalizer(BaseNormalizer):
	def __init__(self, data):
		super().__init__(data)

	def _normalize(self):
		self.std = np.std(self.data)
		self.mean = np.mean(self.data)

	def get_normalized_value(self, vec):
		return (vec - self.mean) / self.std

class NonNormalizer(BaseNormalizer):
	def __init__(self, data):
		super().__init__(data)

	def get_normalized_value(self, vec):
		return vec
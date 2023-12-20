import numpy as np
import json
from datetime import datetime
import random


class BestFeatures(object):
	def __init__(self, variance_explained, max_features, positive=None):
		self.feature_importance = {}
		self.var_explained = variance_explained
		self.max_features = max_features
		self.explained_var_real = 0
		self.__last_importance = {}
		self.__last_pca_data = None
		self.__round = 0
		self.__full_features_amount = 0
		self.__start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		self.positive = positive


	def __get_components_amount(self):
		amount = 0
		self.explained_var_real = 0
		for var in self.__last_pca_data.explained_variance_ratio_:
			if self.explained_var_real >= self.var_explained:
				return amount
			amount += 1
			self.explained_var_real += var
		return self.__last_pca_data.explained_variance_ratio_.size


	def __get_k_biggest_by_index(self, vector):
		if vector.size < self.max_features:
			return range(vector.size)
		return np.argpartition(np.abs(vector), self.max_features)[-self.max_features:]

	def __set_vector(self, vector):
		if self.positive == True:
			return np.array([n if n >= 0 else 0 for n in vector])
		if self.positive == False:
			return np.array([n if n <= 0 else 0 for n in vector])



	def dump_feature_importance(self):
		d = {int(key): val for key, val in self.feature_importance.items()}
		with open("{}_f_imp_{}.txt".format(self.__start_time, self.__round), "w") as a:
			json.dump(d, a)


	def dump_last_calc_feature_importance(self):
		d = {int(key): val for key, val in self.__last_importance.items()}
		with open("{}last_imp_{}.txt".format(self.__start_time, self.__round), "w") as a:
			json.dump(d, a)


	def calc_average_feature_importance(self, pca_data, add_to_all=True):
		#TODO: what if a component[index] is negative
		self.__last_pca_data = pca_data
		self.__last_importance = {}
		self.__round += 1
		size = self.__get_components_amount()
		for i in range(size):
			component = self.__last_pca_data.components_[i]
			self.__full_features_amount = max(self.__full_features_amount, component.size)
			vector = self.__set_vector(component)
			biggest_indecis = self.__get_k_biggest_by_index(vector)
			explained = self.__last_pca_data.explained_variance_ratio_[i]
			for index in biggest_indecis:
				if index not in self.__last_importance:
					self.__last_importance[index] = 0
				self.__last_importance[index] += vector[index] * explained

		if add_to_all:
			self.__add()


	def __add(self):
		for key in set(self.feature_importance.keys()) - set(self.__last_importance.keys()):
			self.feature_importance[key].append(0)

		for key in self.__last_importance.keys():
			if key not in self.feature_importance:
				self.feature_importance[key] = [0] * (self.__round - 1)
			self.feature_importance[key].append(self.__last_importance[key])


	def get_feature_importance(self, dump=False):
		avg_importance = {}
		for key, list_importance in self.feature_importance.items():
			avg_importance[int(key)] = sum(list_importance) / len(list_importance)

		if dump:
			r = int(random.random() * 1000)
			with open("{}_avg_f_imp_{}__{}.txt".format(self.__start_time, self.__round, r), "w") as a:
				json.dump(avg_importance, a)
		return avg_importance

	def get_as_vector(self, is_np=False):
		vector = [0] * self.__full_features_amount
		for key, val in self.get_feature_importance().items():
			vector[key] = val
		return np.array(vector) if is_np else vector





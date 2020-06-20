# import the necessary packages
import numpy as np
import csv, math
import matplotlib.pyplot as plt

class Searcher:
	euc_val = []
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath

	def search(self, queryFeatures, limit = 10):
		# initialize our dictionary of results
		results = {}

		# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)

			# loop over the rows in the index
			for row in reader:
				features = [float(x) for x in row[1:]]
				euc = self.euclidean_distance(features, queryFeatures)

				results[row[0]] = euc

			# close the reader
			f.close()
		results = sorted([(v, k) for (k, v) in results.items()])
        
		return results[:limit]

	def euclidean_distance(self, histA, histB, eps = 1e-10):
		res = math.sqrt(sum([(a - b) ** 2
        for a, b in zip(histA, histB)]))
        #res = d.cdist(histA, histB, 'euclidean')
		return res

	def averagePrecision(self, distance_matrix,classes,retrive_images,relevant_images_DB):
		class_precision = [0]*classes

		i=0
		for distmat in distance_matrix:
			relevant_images=sum(np.floor(np.argsort(distmat)[:retrive_images]/relevant_images_DB)==math.floor(i/relevant_images_DB))
			class_precision[math.floor(i/relevant_images_DB)]+=relevant_images/retrive_images
			l=i
			i=i+1;
			if math.floor(i/relevant_images_DB)!=math.floor(l/relevant_images_DB):
				class_precision[math.floor(l/relevant_images_DB)]/=relevant_images_DB

		class_precision*=100
		average_precision = sum(class_precision)/classes

		return class_precision,average_precision

	def averageRecall(self, distance_matrix,classes,retrive_images,relevant_images_DB):
		class_recall = [0]*classes
		i=0
		for distmat in distance_matrix:
			relevant_images=sum(np.floor(np.argsort(distmat)[:retrive_images]/relevant_images_DB)==math.floor(i/relevant_images_DB))
			class_recall[math.floor(i/relevant_images_DB)]+=relevant_images/relevant_images_DB
			l=i
			i=i+1;
			if math.floor(i/relevant_images_DB)!=math.floor(l/relevant_images_DB):
				class_recall[math.floor(l/relevant_images_DB)]/=relevant_images_DB
		class_recall*=100
		average_recall = sum(class_recall)/classes
		return class_recall,average_recall

	def precision_recall(self, queryFeatures):
		classes = 10
		retrive_images = 10
		relevant_images_DB = 100

		results_euc = {}

		# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)

			# loop over the rows in the index
			for row in reader:
				features = [float(x) for x in row[1:]]
				results_euc[row[0]] = self.euclidean_distance(features, queryFeatures)
			# close the reader
			f.close()
        
		precision_results = []
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images+10,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images+10,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images+20,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images+20,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images+30,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images+30,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images+40,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images+40,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images+50,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images+50,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images+60,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images+60,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images+70,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images+70,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images+80,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images+80,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
        
		euclidean_class_precision,euclidean_average_precision = self.averagePrecision(results_euc,classes,retrive_images+90,relevant_images_DB)
		euclidean_class_recall,euclidean_average_recall = self.averageRecall(results_euc,classes,retrive_images+90,relevant_images_DB)
		precision_results.append(euclidean_average_precision)
        
		print('Euclidean Average Precision: '+str(euclidean_average_precision)+'%')
		print('Euclidean Average Recall: '+str(euclidean_average_recall)+'%\n')
		return precision_results
        
		plt.plot(precision_results)
		plt.show()
# -*- coding: utf-8 -*-

#####
# DUHAMEL Thibault 18026048
# SHI Heng 18171434
###

import numpy as np
import matplotlib.pyplot as plt


class MAPnoyau:
	def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
		"""
		Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

		lamb: coefficiant de régularisation L2
		sigma_square: paramètre du noyau rbf
		b, d: paramètres du noyau sigmoidal
		M,c: paramètres du noyau polynomial
		noyau: rbf, lineaire, polynomial ou sigmoidal
		"""
		self.lamb = lamb
		self.a = None
		self.sigma_square = sigma_square
		self.M = M
		self.c = c
		self.b = b
		self.d = d
		self.noyau = noyau
		self.x_train = None



	def entrainement(self, x_train, t_train):
		"""
		Entraîne une méthode d'apprentissage à noyau de type Maximum a
		posteriori (MAP) avec un terme d'attache aux données de type
		"moindre carrés" et un terme de lissage quadratique (voir
		Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
		contient les entrées (un tableau 2D Numpy, où la n-ième rangée
		correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
		où le n-ième élément correspond à la cible t_n).

		L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
		ou polynomial (spécifié par ''self.noyau'') et dont les parametres
		sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
		et self.M et un poids de régularisation spécifié par ``self.lamb``.

		Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
		l'equation 6.8 du livre de Bishop et garder en mémoire les données
		d'apprentissage dans ``self.x_train``
		"""
		self.x_train = x_train

		N = x_train.shape[0]

		K = np.zeros((N,N))
		for i in range(N):
			for j in range(i+1):
				k_xi_xj = 0
				if self.noyau == "rbf":
					k_xi_xj = np.exp(-np.linalg.norm(x_train[i]-x_train[j])**2/2/self.sigma_square)
				elif self.noyau == "polynomial":
					k_xi_xj = (np.dot(x_train[i],x_train[j])-self.c)**self.M
				elif self.noyau == "sigmoidal":
					k_xi_xj = np.tanh(self.b*np.dot(x_train[i],x_train[j]) + self.d)
				else:
					k_xi_xj = np.dot(x_train[i],x_train[j])
				#K is symetric
				K[i,j] = k_xi_xj
				K[j,i] = k_xi_xj

		#Equation 6.8
		self.a = np.dot(np.linalg.inv(K + self.lamb*np.identity(x_train.shape[0])),t_train)

	def prediction(self, x):
		"""
		Retourne la prédiction pour une entrée representée par un tableau
		1D Numpy ``x``.

		Cette méthode suppose que la méthode ``entrainement()`` a préalablement
		été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
		la prédiction y(x) (équation 6.9).

		NOTE : Puisque nous utilisons cette classe pour faire de la
		classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
		sinon
		"""
		#Equation 6.9
		y = np.dot(self.a,np.dot(self.x_train,x))
		if y>0.5:
			return 1
		return 0

	def erreur(self, t, prediction):
		"""
		Retourne la différence au carré entre
		la cible ``t`` et la prédiction ``prediction``.
		"""
		return (t-prediction)**2

	def validation_croisee(self, x_tab, t_tab):
		"""
		Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
		``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
		``self.lamb`` avec une validation croisée de type "k-fold" où k=1 avec les
		données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
		trouvés, le modèle est entraîné une dernière fois.

		SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
		de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
		de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
		"""
		if self.noyau == "polynomial":
			min_err = float("inf")
			best_M = 2
			best_c = 0
			best_lamb = 0.000000001
			for lamb in [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 2]:
				for M in [0,1,2,3,4,5]:
					for c in [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5]:
						self.M = M
						self.c = c
						self.lamb = lamb
						self.entrainement(x_tab[0:int(0.8*len(x_tab))], t_tab[0:int(0.8*len(t_tab))])
						errors = [self.erreur(t_tab[i],self.prediction(x_tab[i])) for i in range(int(0.8*len(t_tab)),int(len(t_tab)))]
						err = 100*np.sum(errors)/(0.2*len(x_tab))
						if err < min_err:
							min_err = err
							best_M = M
							best_c = c
							best_lamb = lamb
			self.M = best_M
			self.c = best_c
			self.lamb = best_lamb
			print(best_M,best_c,best_lamb,min_err)
			self.entrainement(x_tab,t_tab)

	def affichage(self, x_tab, t_tab):

		# Affichage
		ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
		iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
		iX, iY = np.meshgrid(ix, iy)
		x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
		contour_out = np.array([self.prediction(x) for x in x_vis])
		contour_out = contour_out.reshape(iX.shape)

		plt.contourf(iX, iY, contour_out > 0.5)
		plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
		plt.show()
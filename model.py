# ___________                                ________  ________  
# \__    ___/  ____  _____     _____         \_____  \ \_____  \ 
#   |    |   _/ __ \ \__  \   /     \         /  ____/   _(__  < 
#   |    |   \  ___/  / __ \_|  Y Y  \       /       \  /       \
#   |____|    \___  >(____  /|__|_|  / ______\_______ \/______  /
#                 \/      \/       \/ /_____/        \/       \/
import pandas as pd
import numpy as np

from evaluation_quicky import configuration_evaluation #To compute value function
from systems_specifications import storage_system_specifications #Read battery specs
from plotter import shares_pie_plot

import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

#Plot configuration
sns.set_theme() #use to reset955
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale= 2.5)
plt.rc('figure',figsize=(32,24))
plt.rc('font', size=20)
plt.ioff()
class Producer:
	"""Represents a producer in the configuration"""
	pass 

class DistributionModel:
	"""Benefit distribution model according to Repetto's basic formulation:
	No storage is considered."""

	#----------------------------Nested Participant class----------------------
	class _Participant:
		"""Constructor should not be invoked by user"""
		def __init__(self, participant_number, name, profile_we = None, profile_wd = None):
			"""Parameters:
				player_number: int. 
			"""
			self._profile_wd = profile_wd
			self._profile_we = profile_we

			#Assign player number
			self.participant_number = participant_number
			#Player name
			self.name = name
			#Display dumb menu
			print("Creating Participant {}: {}".format(participant_number, name))
			#Insert PV Power and Battery Size for this user.
			self._pv_size = input("Insert PV Power for this participant: ")
			if self._pv_size == '': 
				self._pv_size = 0
			else:
				self._pv_size = int(self._pv_size)
			
			self._battery_size = 0 #Default to zero in this basic formulation

			#Compute player max power
			self._grid_purchase_max = np.ceil(profile_wd.max())

			#Consumer / Producer flags
			self.is_consumer = (self._grid_purchase_max != 0)
			self.is_producer = (self._pv_size > 0)

			#Compute hourly profiles (production / consumption)
			#np.array format
			#Call to Lorenti's Module - replace np.empty
			self.consumption = np.empty() 
			self.production = np.empty()


	#----------------------------------Private Methods---------------------------------
	def _run_recopt(self, ):
		"""Optimize configuration and return power fluxes"""
		raise NotImplementedError("Add Lorenti's new module")

	def _economic_value(self, shared_energy, grid_feed, PR3 = 42,  CUAF = 8.56 , TP = 110):
		"""Return economic value of shared energy plus energy sales"""
		ritiro_energia = grid_feed/1000 * PR3 #KWh to MWh
		incentivo = shared_energy/1000 * (CUAF + TP)
		return ritiro_energia + incentivo

	def _compute_shares(self, participant):
		"""Compute % shares (suitable for both producers and consumers).
		Runs on hourly basis, returns yearly values"""
		p_production = participant.production
		p_consumption = participant.consumption

		#Phase 1: Producer share
		p_share = np.zeros(len(p_production)) #producer share for this participant 
		if participant.is_producer:
			for ix, single, total in enumerate(zip(p_production, self._total_production)):
				#Look before you leap
				if total == 0:
					continue #p_share remain zero
				p_share[ix] = single/total

		#Phase 2: Consumer share
		c_share = np.zeros(len(p_production)) #consumer share for this participant 
		if participant.is_consumer:
			for ix, single, total in enumerate(zip(p_consumption, self._total_consumption)):
				#Look before you leap
				if total == 0:
					continue #p_share remain zero
				c_share[ix] = single/total			

		

	def _create_participants(self):
		"""Create participants to REC.
		Returns:
		List of _Participant objects
		
		"""
		basepath= Path(__file__).parent
		datapath = basepath / 'Data'
		data_folders = os.listdir(datapath)
		participant_list = [] #Store _Participant objects
		for ix, folder in enumerate(data_folders):
			#Workday
			wd_path = datapath / folder / "consumption_profiles_month_wd.csv"
			wd_data = pd.read_csv(wd_path,
								  sep = ';',
								  decimal= ',',
								  ).dropna().values[:,1:]
			#Weekend
			we_path = datapath / folder / "consumption_profiles_month_we.csv"
			we_data = pd.read_csv(we_path,
								  sep = ';',
								  decimal= ',',
								  ).dropna().values[:,1:]

			#Instantiate new player
			new = self._Participant(ix, folder, wd_data, we_data) #folder stays for participant name
			participant_list.append(new)

		return player_list
	#----------------Public Methods------------------
	def run(self):
		"""Run distribution model, return shares"""
		pass

	def __init__(self, eta = 0.5):
		"""Create an instance of the model.
		Parameters.
		eta: float in [0,1]. Share of Benefit for producers.
		"""
		self.eta = eta
		#Create participants
		self.participants = self._create_participants()
		#Run recopt and get quantities
		self._run_recopt() 
		self._total_value = np.empty(8760) #Economic value on hourly basis
		#aggregate values over all participants (hourly arrays)
		self._total_production = np.sum([p.production for p in self.participants])
		self._total_consumption = np.sum([p.consumption for p in self.participants])



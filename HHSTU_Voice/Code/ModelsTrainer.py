import os
import pickle
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture
from FeaturesExtractor import FeaturesExtractor
from scipy.io.wavfile import read

class trainer():
    def mt(self,inputwav):
        warnings.filterwarnings("ignore")

        features = np.asarray(())

        features_extractor = FeaturesExtractor()

        sr, audio = read(inputwav)
        vector                 = features_extractor.accelerated_get_features_vector(sr, audio, 8000)
        if features.size == 0:
            features = vector
        else:
            try:
                features = np.vstack((features, vector))
            except:
                print("错误: Shape mismatch")

        gmm = GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag', n_init = 3)
        gmm.fit(features)

        picklefile = inputwav.split(".")[0] + ".gmm"
        with open(picklefile, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
qn =trainer()
qn.mt("李澳.wav")


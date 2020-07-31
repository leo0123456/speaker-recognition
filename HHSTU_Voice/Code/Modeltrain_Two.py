
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
from scipy.io.wavfile import read
from GMM_EM import Gmmly

class trainer():
    def mt(self,inputwav):
        warnings.filterwarnings("ignore")

        features = np.asarray(())

        features_extractor = FeaturesExtractor()
        GMM_two = Gmmly()

        sr, audio = read(inputwav)
        vector                 = features_extractor.accelerated_get_features_vector(sr, audio, 8000)
        if features.size == 0:
            features = vector
        else:
            try:
                features = np.vstack((features, vector))
            except:
                print("错误: Shape mismatch")


        gmm = GMM_two.gmm_em(features,2,10)

        picklefile = inputwav.split(".")[0] + ".gmm"
        with open(picklefile, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
qn =trainer()
qn.mt("李澳.wav")


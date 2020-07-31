import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
from scipy.io.wavfile import read


class spk_id():
    def person_identifer(self):
        warnings.filterwarnings("ignore")


        modelpath = "SpeakerModels/"



        db = {}
        for fname in [fname for fname in os.listdir(modelpath) if fname.endswith('.gmm')]:
            speaker     = fname.split('.')[0]
            model       = pickle.load( open(os.path.join(modelpath, fname), 'rb'))
            db[speaker] = model


        error, total_sample = 0, 0


        features_extractor = FeaturesExtractor()
        sr, audio = read("output.wav")
        vector                 = features_extractor.accelerated_get_features_vector(sr, audio, 8000)


        if vector.shape != (0,):
            print(vector.shape)
            total_sample      += 1
            log_likelihood     = {}
            m                  = {}
            for speaker, model in db.items():
                gmm                     = model
                scores                  = np.array(gmm.score(vector))
                log_likelihood[speaker] = round(scores.sum(), 3)
                m[speaker]              = scores

            max_log_likelihood = max(log_likelihood.values())
            keys, values       = list(log_likelihood.keys()), list(log_likelihood.values())
            winner             = keys[values.index(max_log_likelihood)]

            return winner








import numpy as np
from sklearn import preprocessing
from python_speech_features import mfcc
from python_speech_features import delta


class FeaturesExtractor:
    def __init__(self):
        pass

    def extract_features(self, audio, rate):

        mfcc_feature = mfcc(
                            audio,
                            rate,
                            winlen = 0.025,
                            winstep = 0.01,
                            numcep = 20,
                            nfilt = 30,
                            nfft = 512,
                            appendEnergy = True)


        mfcc_feature  = preprocessing.scale(mfcc_feature)
        deltas        = delta(mfcc_feature, 2)
        double_deltas = delta(deltas, 2)
        combined      = np.hstack((mfcc_feature, deltas, double_deltas))
        return combined

    def accelerated_get_features_vector(self, input_wave_file, audio, sound_rate):

        try   :
            return self.extract_features(audio, sound_rate)

        except:
            print("不能找到文件", input_wave_file.split('.')[0])
            return np.array([])

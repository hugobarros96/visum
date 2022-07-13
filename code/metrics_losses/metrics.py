from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import cv2
class Error:
    '''
    Calculates specific errors, between 2 inputs (grounth truth and predictions)
    '''
    

    def __init__(self,true,pred):
        assert np.shape(true)==np.shape(pred), "the inputs must be an 1Darray/list of equal size"
        self.true=true
        self.pred=pred
    
    def get_mse(self):
       return mean_squared_error(self.true, self.pred)
    
    def get_mae(self):
        return  mean_absolute_error(self.true, self.pred)

    def get_rmse(self):
        return np.sqrt(mean_absolute_error(self.true, self.pred))

    def get_rmse_log(self):
        return np.sqrt(np.mean(np.power(np.log1p(self.pred)-np.log1p(self.true), 2)))

    def get_tresh_accuracy(self):
        t=self.true[self.true>0]
        p=self.pred[self.pred>0]
        thresh=[max((t[i]/p[i]),(p[i]/t[i])) for i in range(len(self.true))]
        a1 = np.mean(np.asarray(thresh) < 1.25)
        a2 = np.mean(np.asarray(thresh) < 1.25 ** 2)
        a3 = np.mean(np.asarray(thresh) < 1.25 ** 3)
        return (a1,a2,a3)

    def get_AP(self):
        precision = 
        recall = 
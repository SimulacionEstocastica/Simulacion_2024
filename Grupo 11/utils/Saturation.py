import numpy as np
import matplotlib.pyplot as plt

def plot_hist_uni(diff, p, q, N, savepath):
    print(p, q, type(p), type(q))
    if q.shape != () and q.shape[0] != 0:
        C = N*(p-q)
    else:
        C = N*p
    f, ax = plt.subplots()
    plt.hist(diff, bins = min(diff.shape[0], 20))
    plt.axvline(x = C, color = 'r', label = "Expectation")
    plt.xlabel("Total movement during the simulation")
    plt.ylabel("Number of particules")
    plt.title("Saturation evalutation")
    plt.text(0.25,0.9 ,"Mean: {:.1f} \n Variation to expectation: {:.1f} %".format(np.mean(diff), 100*(1 - np.mean(diff)/C)),
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax.transAxes)
    plt.legend()
    plt.savefig(savepath, dpi = 300)
    plt.close()



class SaturationEstimator:
    def __init__(self, case, ley_number_particles, p, q = []):
        self.p = np.array(p)
        self.q = np.array(q)
        self.case = case
        self.ley_number_particles = ley_number_particles

        self.call_func = self._choose_func()

    def _saturation_uni(self, pos_record, savepath, **kwargs):
        if self.ley_number_particles == "stable":
            diff = pos_record[-1] - pos_record[0]
            plot_hist_uni(diff, self.p, self.q, len(pos_record), savepath)
        else:
            raise NotImplementedError

    def _saturation_multi(self,pos_record, savepath, **kwargs):
        raise NotImplementedError
    
    def _saturation_multi_HD(self, pos_record,savepath, **kwargs):
        raise NotImplementedError

    def _choose_func(self):
        match self.case:
            case "uni":
                return self._saturation_uni
            case "multi":
                return self._saturation_multi
            case "multi_HD":
                return self._saturation_multi_HD
            
    def __call__(self, pos_record, savepath):
        return self.call_func(pos_record, savepath)
    
            
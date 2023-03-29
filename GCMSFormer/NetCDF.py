import scipy.io.netcdf as nc
from scipy.sparse import coo_matrix
from numpy import zeros,hstack,linspace
import numpy as np
from matplotlib.ticker import  FormatStrFormatter
from pylab import figure,show,plot
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

class netcdf_reader:
    def __init__(self,filename,bmmap=True):
        self.filename=filename
        self.f = nc.netcdf_file(filename,'r',mmap=bmmap)
        self.mass_values            = self.f.variables['mass_values']
        self.intensity_values       = self.f.variables['intensity_values']
        self.scan_index             = self.f.variables['scan_index']
        self.total_intensity        = self.f.variables['total_intensity'];
        self.scan_acquisition_time  = self.f.variables['scan_acquisition_time'];
        self.mass_max = np.max(self.f.variables['mass_range_max'].data);
        self.mass_min = np.max(self.f.variables['mass_range_min'].data);

    def mz(self,n):
    
        scan_index_end=hstack((self.scan_index.data,np.array([len(self.intensity_values.data)],dtype=int)))
        ms={}
        inds=range(scan_index_end[n],scan_index_end[n+1]);
        ms['mz']=self.mass_values[inds];
        ms['intensity']=self.intensity_values[inds];
        return ms;

    def mz_rt(self,t):
    
        scan_index_end=hstack((self.scan_index.data,np.array([len(self.intensity_values.data)],dtype=int)))
        ms={}
        tic=self.tic();
        rt=tic['rt']
        n=np.searchsorted(rt, t)
        inds=range(scan_index_end[n],scan_index_end[n+1]);
        ms['mz']=self.mass_values[inds];
        ms['intensity']=self.intensity_values[inds];
        return ms;
    
    def rt(self,rt_start,rt_end):
        indmin, indmax = np.searchsorted(self.tic()['rt'], (rt_start, rt_end))
        rt=self.tic()['rt'][indmin:indmax+1]
        return rt;

    def tic(self):
        nsize=len(self.total_intensity.data);
        tic={}
        tic['rt']=self.scan_acquisition_time.data/60.0;
        tic['intensity']=self.total_intensity.data
        return tic;
    
    def mat_rt(self,rt_start,rt_end):
        indmin, indmax = np.searchsorted(self.tic()['rt'], (rt_start, rt_end))
        rt=self.tic()['rt'][indmin:indmax+1]
        mass_max = np.max(self.f.variables['mass_range_max'].data);
        mass_min = np.max(self.f.variables['mass_range_min'].data);
        mz=np.linspace(mass_min, mass_max,num=int(mass_max-mass_min+1))
        return self.mat(indmin,indmax)

    def mat(self,imin,imax):
        f = nc.netcdf_file(self.filename,'r',mmap=False)
        mass_values            = f.variables['mass_values']
        intensity_values       = f.variables['intensity_values']
        scan_index             = f.variables['scan_index']
        scan_index_end=np.hstack((scan_index.data,np.array([len(intensity_values.data)],dtype=int)))
        mass_max = np.max(f.variables['mass_range_max'].data);
        mass_min = np.max(f.variables['mass_range_min'].data);
        rg=np.linspace(mass_min-0.5, mass_max+0.5,num=int(mass_max-mass_min+2))
        c=int(imax-imin+1)
        r=int(mass_max-mass_min+1)
        z=int(np.ceil(1/np.min(abs(np.diff(mass_values.data)))))
        mo=np.zeros((r,c,z))
        for j in range(imin,imax+1):
            mz=mass_values[scan_index_end[j]:scan_index_end[j+1]]
            ms=intensity_values[scan_index_end[j]:scan_index_end[j+1]]
            inds=np.searchsorted(mz, rg)
            for i in range(0,r):
                mo[i,j-imin,0:(inds[i+1]-inds[i])]=ms[inds[i]:inds[i+1]]
        return np.sum(mo,2)

def plot_ms(ms):
    fig = figure()
    ax = fig.add_subplot(111)
    ax.vlines(ms['mz'], zeros((len(ms['mz']),)),ms['intensity'], color='k', linestyles='solid')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%3.0f'))
    show()

def plot_tic(tic):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(tic['rt'],tic['intensity'])
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    y_major_locator=MultipleLocator(300000)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xlabel('Retention Time', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('TIC',fontsize=12)
    plt.show()
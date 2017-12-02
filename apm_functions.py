from apm import *

def sim_init(s,a):
   apm(s,a,'clear all')
   apm_load(s,a,'process.apm')
   csv_load(s,a,'process.csv')
   apm_option(s,a,'nlc.imode',4)
   apm_option(s,a,'nlc.nodes',3)
   apm_info(s,a,'SV','y')
   apm_info(s,a,'FV','u')
   apm_info(s,a,'FV', 'k')
   apm_option(s,a,'u.fstatus',1)
   apm_option(s,a,'K.fstatus',1)
   msg = 'Successful simulator initialization'
   return msg

def mpc_init(s,a):
   apm(s,a,'clear all')
   apm_load(s,a,'model.apm')
   csv_load(s,a,'data.csv')
   apm_option(s,a,'nlc.imode',6)
   apm_option(s,a,'nlc.nodes',3)
   apm_option(s,a,'nlc.web_plot_freq',1)
   apm_info(s,a,'FV','K')
   apm_info(s,a,'FV','tau')
   apm_info(s,a,'MV','u')
   apm_info(s,a,'CV','y')
   # status, whether the optimizer can use it
   apm_option(s,a,'K.status',0)
   apm_option(s,a,'tau.status',0)
   apm_option(s,a,'u.status',1)
   apm_option(s,a,'y.status',1)
   # feedback status
   apm_option(s,a,'K.fstatus',1)
   apm_option(s,a,'tau.fstatus',1)
   apm_option(s,a,'u.fstatus',0)
   apm_option(s,a,'y.fstatus',1)
   # constraints
   apm_option(s,a,'u.upper',100)
   apm_option(s,a,'u.lower',0)
   # reference trajectory tuning
   apm_option(s,a,'nlc.traj_init',2)
   apm_option(s,a,'nlc.traj_open',0.5)
   apm_option(s,a,'y.tau',12)
   msg = 'Successful controller initialization'
   return msg
   
def sim(s,a,u,K):
   apm_meas(s,a,'u',u)
   apm_meas(s,a,'k',K)
   apm(s,a,'solve')
   y = apm_tag(s,a,'y.model')
   return y
   
def mpc(s,a,inputs):
   sp = inputs[0]
   y_meas = inputs[1]
   k_pred = inputs[2]
   apm_meas(s,a,'y',y_meas)
   apm_meas(s,a,'k',k_pred)
#   apm_meas(s,a,'tau',tau_pred)
   sphi = sp + 0.1
   splo = sp - 0.1
   apm_option(s,a,'y.sphi',sphi)
   apm_option(s,a,'y.splo',splo)
   apm_option(s,a,'y.sp',sp)
   apm(s,a,'solve')
   u = apm_tag(s,a,'u.newval')
   y_pred5 = apm_tag(s,a,'y.pred[5]')
   return u, y_pred5
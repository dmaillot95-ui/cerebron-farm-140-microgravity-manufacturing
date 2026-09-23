import json,hashlib,pathlib,math,platform
import warp as wp
wp.init();N=4096;dt=0.001;steps=1000;omega=2.0;x0=0.01;v0=0.0
@wp.kernel
def step(x:wp.array(dtype=float),v:wp.array(dtype=float),dt:float,w:float):
 i=wp.tid();a=-(w*w)*x[i];v[i]=v[i]+a*dt;x[i]=x[i]+v[i]*dt
x=wp.array([x0]*N,dtype=float,device="cpu");v=wp.array([v0]*N,dtype=float,device="cpu")
for _ in range(steps):wp.launch(step,dim=N,inputs=[x,v,dt,omega],device="cpu")
xa=x.numpy();va=v.numpy();t=dt*steps;xe=x0*math.cos(omega*t)+(v0/omega)*math.sin(omega*t);ve=-x0*omega*math.sin(omega*t)+v0*math.cos(omega*t);mx=float(xa.mean());mv=float(va.mean());ex=abs(mx-xe);ev=abs(mv-ve);ok=ex<5e-5 and ev<1e-4
out={"farm":140,"engine":"NVIDIA Warp","warp_version":getattr(wp,"__version__","unknown"),"test":"DYNAMIC_INTEGRATOR_ANALYTIC_REPRODUCTION","elements":N,"steps":steps,"updates":N*steps,"dt_s":dt,"omega_rad_s":omega,"warp_x":mx,"analytic_x":xe,"x_abs_error":ex,"warp_v":mv,"analytic_v":ve,"v_abs_error":ev,"status":"REFERENCE_REPRODUCED_OK" if ok else "FAIL","scope":"REAL_WARP_DYNAMIC_NUMERICAL_REPRODUCTION_AGAINST_INDEPENDENT_ANALYTIC_OSCILLATOR_NOT_PHYSICAL_EXPERIMENT","python":platform.python_version()};raw=json.dumps(out,sort_keys=True).encode();out["result_sha256"]=hashlib.sha256(raw).hexdigest();pathlib.Path("artifacts").mkdir(exist_ok=True);pathlib.Path("artifacts/f140_warp_reference_v3.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out));raise SystemExit(0 if ok else 1)

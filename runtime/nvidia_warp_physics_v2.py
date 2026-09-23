import json,hashlib,pathlib,platform
import warp as wp
wp.init()
N=2048;dt=0.002;steps=500;g=0.001;k=4.0;m=1.0
@wp.kernel
def step(x:wp.array(dtype=float),v:wp.array(dtype=float),dt:float,g:float,k:float,m:float):
 i=wp.tid();a=-g-(k/m)*x[i];v[i]=v[i]+a*dt;x[i]=x[i]+v[i]*dt
x=wp.array([0.01]*N,dtype=float,device="cpu");v=wp.zeros(N,dtype=float,device="cpu")
for _ in range(steps): wp.launch(step,dim=N,inputs=[x,v,dt,g,k,m],device="cpu")
xa=x.numpy();va=v.numpy();mean_x=float(xa.mean());mean_v=float(va.mean());energy=float((0.5*m*(va*va)+0.5*k*(xa*xa)+m*g*xa).mean());ok=all(map(lambda q:abs(q)<10,[mean_x,mean_v,energy])) and energy>0
out={"farm":140,"engine":"NVIDIA Warp","warp_version":getattr(wp,"__version__","unknown"),"device":"cpu","test":"MICROGRAVITY_MASS_SPRING_2048","particles":N,"steps":steps,"dt_s":dt,"mean_x_m":mean_x,"mean_v_m_s":mean_v,"mean_specific_energy_j":energy,"status":"PHYSICS_KERNEL_OK" if ok else "FAIL","scope":"DYNAMIC_WARP_PHYSICS_KERNEL_NOT_MULTIPHASE_CFD_OR_PHYSICAL_EXPERIMENT","python":platform.python_version()};raw=json.dumps(out,sort_keys=True).encode();out["result_sha256"]=hashlib.sha256(raw).hexdigest();pathlib.Path("artifacts").mkdir(exist_ok=True);pathlib.Path("artifacts/f140_nvidia_warp_physics_v2.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out));raise SystemExit(0 if ok else 1)

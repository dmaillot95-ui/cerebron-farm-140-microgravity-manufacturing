import json,hashlib,pathlib,platform
try:
 import warp as wp
except Exception as e:
 print(json.dumps({"farm":140,"status":"FAIL","reason":"WARP_IMPORT","error":str(e)}));raise
wp.init()
@wp.kernel
def bond_kernel(rho:wp.array(dtype=float),g:wp.array(dtype=float),L:wp.array(dtype=float),sigma:wp.array(dtype=float),out:wp.array(dtype=float)):
 i=wp.tid();out[i]=rho[i]*g[i]*L[i]*L[i]/sigma[i]
rho=wp.array([1000.0],dtype=float,device="cpu");g=wp.array([0.001],dtype=float,device="cpu");L=wp.array([0.01],dtype=float,device="cpu");sigma=wp.array([0.072],dtype=float,device="cpu");outa=wp.zeros(1,dtype=float,device="cpu");wp.launch(bond_kernel,dim=1,inputs=[rho,g,L,sigma,outa],device="cpu");value=float(outa.numpy()[0]);ref=1000*.001*.01*.01/.072;delta=abs(value-ref);ok=delta<1e-7
res={"farm":140,"engine":"NVIDIA Warp","warp_version":getattr(wp,"__version__","unknown"),"device":"cpu","test":"BOND_NUMBER_KERNEL","warp_value":value,"python_reference":ref,"absolute_delta":delta,"status":"EXTERNAL_ENGINE_REPRODUCTION_OK" if ok else "FAIL","scope":"REAL_WARP_KERNEL_EXECUTION_NOT_MULTIPHASE_CFD_OR_PHYSICAL_VALIDATION","python":platform.python_version()};raw=json.dumps(res,sort_keys=True).encode();res["result_sha256"]=hashlib.sha256(raw).hexdigest();pathlib.Path("artifacts").mkdir(exist_ok=True);pathlib.Path("artifacts/f140_nvidia_warp.json").write_text(json.dumps(res,indent=2)+"\n");print(json.dumps(res));raise SystemExit(0 if ok else 1)

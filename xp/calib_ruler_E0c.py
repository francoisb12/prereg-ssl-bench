# Calibrage a posteriori de la regle E0c (acf_at_rf, ell_acf) sur des champs synthetiques.
# Ni donnees ni modele. Lance le 2026-09-16 apres E-003 ; sortie archivee dans
# results/E0c/calib_ruler_synthetic.txt. Sur le portable : 308 s CPU et 3,3 Go de RAM,
# NE PAS relancer localement ; sur Colab GPU, quelques secondes :
#   %cd /content/drive/MyDrive/francois/ubergang_xp && python xp/calib_ruler_E0c.py
# Calibration de l'instrument E0c sur des champs synthetiques (CPU, quelques secondes).
# Aucune donnee, aucun modele : on demande seulement ce que acf_at_rf et ell_acf
# rendent sur (a) un champ blanc, (b) un grain a ~3 px, (c) un grain module par une
# enveloppe coherente a ~16 px, (d) enveloppe a ~32 px. Champs au carre, comme se.
import sys, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'xp'); sys.path.insert(0,'.')
import xp_E_router as ER, xp_E0c_coco as EC
torch.manual_seed(0)
B,H=24,128; patch=8; win_mult=8; rf=23.0
def gauss_blur(x, sigma):
    k=int(6*sigma)|1; t=torch.arange(k)-k//2; g=torch.exp(-t.float()**2/(2*sigma**2)); g=(g/g.sum()).to(x.device)
    x=F.pad(x.unsqueeze(1),(k//2,)*4,mode='reflect')
    x=F.conv2d(x,g.view(1,1,1,k)); x=F.conv2d(x,g.view(1,1,k,1)); return x.squeeze(1)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
n=torch.randn(B,H,H,device=dev)
fields={
 'a) blanc, n^2': n**2,
 'b) grain 1.5px, (blur n)^2': gauss_blur(n,1.5)**2,
 'c) grain x enveloppe 16px': (n*(1+2*gauss_blur(torch.randn(B,H,H,device=dev),16)*16))**2,
 'd) grain x enveloppe 32px': (n*(1+2*gauss_blur(torch.randn(B,H,H,device=dev),32)*32))**2,
 'e) blanc + blob (moitie des images)': n**2 + torch.cat([3*gauss_blur(torch.randn(B//2,H,H,device=dev),12)**2*100, torch.zeros(B-B//2,H,H,device=dev)],0),
}
valid=torch.ones(B,H,H,device=dev)
t0=time.time()
for name,f in fields.items():
    f=f.float()
    st=ER.field_statistics(f,valid,patch,chunk=32,win_mult=win_mult)
    a=EC.radial_acf_at(f,patch,win_mult,radii=[rf],chunk=32)[rf]
    ell=st["ell_acf"].cpu().numpy().ravel(); ac=a.cpu().numpy().ravel(); en=st["acf_energy"].cpu().numpy().ravel()
    print(f"{name:38s} acf_at_rf mean={ac.mean():+.4f} std={ac.std():.4f} q95={np.quantile(ac,0.95):+.4f} | ell mean={ell.mean():.2f} q95={np.quantile(ell,0.95):.2f} frac>rf={(ell>rf).mean():.3f} | acf_energy={en.mean():.3f}")
print(f"[{time.time()-t0:.1f}s CPU]")

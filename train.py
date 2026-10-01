"""
Entrenamiento (PC de mesa con GPU).
Uso:  python train.py --data ruta/a/shipsnet/shipsnet
  (carpeta con los PNG del dataset, nombres tipo  1__20180708_180908_0f47__-118.3_33.7.png)

Genera:  models/ (cnn_fold*.pt, svm.joblib, meta.json), results.json, blind_test/ (+ blind_test_labels.csv)
"""
import time, math, argparse, glob, os, json, shutil, random, csv, math
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, joblib
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--extra", default=None, help="carpeta de make_patches.py (dataset adicional)")
ap.add_argument("--svm_fast", action="store_true", help="SVM sin ablación ni grid (mucho más rápido)")
ap.add_argument("--prof_dir", default=None, help="carpeta con las imágenes de ejemplo")
ap.add_argument("--prof_csv", default=None, help="CSV exportado por la app (columnas filename,real)")
ap.add_argument("--prof_repeat", type=int, default=8)
ap.add_argument("--aug", choices=["mild", "strong"], default="mild")
ap.add_argument("--svm_aug", action="store_true", help="copia aumentada para SVM (más lento)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", dev); T_START = time.time()
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------------ Datos
files = sorted(glob.glob(os.path.join(args.data, "*.png")))
assert files, "No se encontraron PNG en --data"
X = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files])
y = np.array([int(os.path.basename(f).split("__")[0]) for f in files])
scenes = np.array([os.path.basename(f).split("__")[1] for f in files])   # grupo = escena satelital
print(f"{len(X)} imágenes | barcos={y.sum()} no-barcos={(1 - y).sum()}")

# Test ciego 15%: split POR ESCENA para evitar fuga de información entre recortes de la misma imagen
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
dev_idx, test_idx = next(gss.split(X, y, scenes))
Xd, yd, gd = X[dev_idx], y[dev_idx], scenes[dev_idx]
Xt, yt = X[test_idx], y[test_idx]
is_extra = np.zeros(len(Xd), bool)
if args.extra:                                    # dataset adicional: SOLO entra a desarrollo, nunca al test ciego
    ef = sorted(glob.glob(os.path.join(args.extra, "*.png")))
    Xe = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in ef])
    ye = np.array([int(os.path.basename(f).split("__")[0]) for f in ef])
    ge = np.array(["x_" + os.path.basename(f).split("__")[1] for f in ef])
    Xd, yd, gd = np.concatenate([Xd, Xe]), np.concatenate([yd, ye]), np.concatenate([gd, ge])
    is_extra = np.r_[is_extra, np.ones(len(Xe), bool)]
    print(f"Extra: {len(Xe)} parches (barcos={ye.sum()})")
is_prof = np.zeros(len(Xd), bool)
if args.prof_dir:                                 
    rows = [r for r in csv.DictReader(open(args.prof_csv)) if (r.get("real") or r.get("label") or "") != ""]
    Xp = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(args.prof_dir, r["filename"])), cv2.COLOR_BGR2RGB), (80, 80), interpolation=cv2.INTER_AREA) for r in rows])
    yp = np.array([int(r.get("real") or r.get("label")) for r in rows]); gp = np.array([f"p_{i}" for i in range(len(rows))])
    R = args.prof_repeat
    Xd, yd, gd = np.concatenate([Xd] + [Xp] * R), np.concatenate([yd] + [yp] * R), np.concatenate([gd] + [gp] * R)
    is_extra = np.r_[is_extra, np.zeros(len(Xp) * R, bool)]; is_prof = np.r_[is_prof, np.ones(len(Xp) * R, bool)]
    print(f"Prof: {len(Xp)} imágenes x{R} (barcos={yp.sum()})")

os.makedirs("blind_test", exist_ok=True)
with open("blind_test_labels.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["filename", "label"])
    for n, i in enumerate(np.random.permutation(test_idx)):
        name = f"img_{n:04d}.png"                    # nombre anónimo: no filtra la etiqueta
        shutil.copy(files[i], os.path.join("blind_test", name)); wr.writerow([name, int(y[i])])
print(f"Dev={len(Xd)}  Test ciego={len(Xt)} (guardado en blind_test/)")

folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=args.seed).split(Xd, yd, gd))

def metrics(yt_, p, thr=0.5):
    yp = (p >= thr).astype(int)
    return dict(acc=accuracy_score(yt_, yp), precision=precision_score(yt_, yp, zero_division=0),
                recall=recall_score(yt_, yp), f1=f1_score(yt_, yp), cm=confusion_matrix(yt_, yp).tolist())

res = {}
# ------------------------------------------------------------------ Descriptores clásicos + SVM
print("\n[1/3] Descriptores + SVM (con copia aumentada: zoom/rotación/color)")
rng = np.random.default_rng(args.seed); N = len(Xd)
feats = lambda A_: Parallel(n_jobs=-1)(delayed(feature_groups)(im) for im in A_)
stack = lambda fl: {k: np.stack([d[k] for d in fl]) for k in ("hog", "color", "lbp")}
G0, Gt = stack(feats(Xd)), stack(feats(Xt))
if args.svm_aug:
    G1 = stack(feats(np.stack([augment_np(im, rng) for im in Xd])))
    G = {k: np.concatenate([G0[k], G1[k]]) for k in G0}; y2 = np.concatenate([yd, yd])
    folds2 = [(np.concatenate([tr, tr + N]), va) for tr, va in folds]     # val = solo originales (sin fuga)
else: G, y2, folds2 = G0, yd, folds
cat = lambda D, ks: np.concatenate([D[k] for k in ks], 1)
base = lambda **kw: make_pipeline(StandardScaler(), PCA(256, svd_solver="randomized", random_state=0), SVC(kernel="rbf", class_weight="balanced", **kw))
res["ablacion_descriptores_svm_cv5"] = {}
T0 = time.time(); print(f"  descriptores listos ({T0 - T_START:.0f}s)", flush=True)
for ks in ([] if args.svm_fast else [("hog",), ("color",), ("lbp",), ("hog", "color"), ("hog", "color", "lbp")]):
    s = cross_val_score(base(C=10, gamma="scale"), cat(G, ks), y2, cv=folds2, n_jobs=-1)
    res["ablacion_descriptores_svm_cv5"]["+".join(ks)] = [float(s.mean()), float(s.std())]
    print(f"  {'+'.join(ks):18s} acc CV = {s.mean():.4f} ± {s.std():.4f}")
ALL = ("hog", "color", "lbp")
if args.svm_fast:
    bp = {"C": 10, "gamma": "scale"}; res["hiperparametros_svm"] = {"mejor": bp, "nota": "fast: sin grid"}
    print("  SVM fast: C=10 gamma=scale")
else:
    grid = GridSearchCV(base(), {"svc__C": [10, 100], "svc__gamma": [1e-4, 3e-4]}, cv=folds2, n_jobs=-1)
    grid.fit(cat(G, ALL), y2)
    bp = {k.split("__")[1]: v for k, v in grid.best_params_.items()}
    res["hiperparametros_svm"] = {"mejor": bp, "acc_cv": float(grid.best_score_),
        "grid": [{"params": p, "acc": float(m)} for p, m in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"])]}
    print("  Mejor:", bp, f"acc CV={grid.best_score_:.4f}")
print(f"  SVM: 5 folds en paralelo + modelo final ({time.time() - T_START:.0f}s)", flush=True)
XA = cat(G, ALL)
def _fit_dec(tr, va):
    return va, base(**bp).fit(XA[tr], y2[tr]).decision_function(XA[va])
svm_dec = np.zeros(N)
for va, d in Parallel(n_jobs=min(5, os.cpu_count() or 1))(delayed(_fit_dec)(tr, va) for tr, va in folds2): svm_dec[va] = d
platt = LogisticRegression().fit(svm_dec.reshape(-1, 1), yd)          # calibración de Platt con decisiones fuera de fold
svm_oof = platt.predict_proba(svm_dec.reshape(-1, 1))[:, 1]
svm = SvmProb(base(**bp).fit(XA, y2), platt)
svm_test = svm.predict_proba(cat(Gt, ALL))[:, 1]
joblib.dump(svm, "models/svm.joblib")

# ------------------------------------------------------------------ CNN (5 folds => ensamble)
print(f"\n[2/3] CNN + validación cruzada (SVM terminó a los {time.time() - T_START:.0f}s) -- ahora usa la GPU", flush=True)
torch.backends.cudnn.benchmark = True
mean, std = [0.0] * 3, [1.0] * 3          # compat: la normalización es por imagen dentro del modelo
Xd_t = to_tensor(Xd).to(dev)

K, ZR = (1.0, (.6, 2.4)) if args.aug == 'strong' else (.35, (.85, 1.2))

def gpu_augment(x):
    """Geometría (zoom 0.6x-2.4x, rotación libre, traslación, flip) + color (saturación, tinte, gamma,
    contraste) + nitidez/desenfoque + ruido. Simula otros sensores y resoluciones."""
    B, d = x.size(0), x.device
    U = lambda lo, hi, *s: torch.empty(B, *s, device=d).uniform_(lo, hi)
    zoom, ang = torch.exp(U(math.log(ZR[0]), math.log(ZR[1]))), U(0, 2 * math.pi)
    c, s_ = torch.cos(ang) / zoom, torch.sin(ang) / zoom
    th = torch.stack([torch.stack([c, -s_, U(-.15, .15)], 1), torch.stack([s_, c, U(-.15, .15)], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
    if random.random() < .5: x = x.flip(3)
    g = x.mean(1, keepdim=True); x = g + (x - g) * U(1 - .5 * K, 1 + .6 * K, 1, 1, 1)
    x = (x * U(1 - .2 * K, 1 + .2 * K, 3, 1, 1)).clamp(0, 1) ** U(1 - .3 * K, 1 + .4 * K, 1, 1, 1)
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * U(1 - .4 * K, 1 + .5 * K, 1, 1, 1) + m
    bl = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)
    x = x + U(-K, 1.2 * K, 1, 1, 1) * (x - bl)                   # <0 desenfoca, >0 enfoca
    return (x + torch.randn_like(x) * U(0, .03, 1, 1, 1)).clamp(0, 1)

def train_cnn(tr, va, seed):
    torch.manual_seed(seed)
    xt_, yt_ = Xd_t[tr], torch.tensor(yd[tr], dtype=torch.float32, device=dev); xv = Xd_t[va]
    net = ShipNet().to(dev); bs = 64; spe = len(tr) // bs
    pw = float((1 - yt_).sum() / yt_.sum().clamp(min=1))      # peso de la clase barco = balance real del fold
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.epochs * spe)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda"); hist = []
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]; yb = yt_[idx]
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                l = nn.functional.binary_cross_entropy_with_logits(net(gpu_augment(xt_[idx])), yb * .95 + .025, reduction="none")
            loss = (l * (1 + yb * (pw - 1))).mean()               # pérdida balanceada por clase
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        net.eval()
        with torch.no_grad(): pv = torch.sigmoid(net(xv)).cpu().numpy()
        hist.append(float(((pv >= .5) == yd[va]).mean()))
    return net.cpu(), hist

cnn_oof = np.zeros(len(Xd)); models_, hists = [], []
for k, (tr, va) in enumerate(folds):
    net, h = train_cnn(tr, va, args.seed + k)
    cnn_oof[va] = predict_cnn([net], Xd[va], device=dev)
    torch.save(net.state_dict(), f"models/cnn_fold{k}.pt"); models_.append(net); hists.append(h)
    print(f"  [{time.time() - T_START:.0f}s] fold {k}: acc val (TTA) = {((cnn_oof[va] >= .5) == yd[va]).mean():.4f}")
res["cnn_cv5"] = {"acc_por_fold": [float(((cnn_oof[va] >= .5) == yd[va]).mean()) for _, va in folds],
                  "curvas_val_por_epoca": hists}

# ------------------------------------------------------------------ Ensamble y test ciego
print("\n[3/3] Ensamble y test ciego")
ws = np.linspace(0, 1, 21)
accs = [accuracy_score(yd, ((w * cnn_oof + (1 - w) * svm_oof) >= .5).astype(int)) for w in ws]
w = float(ws[int(np.argmax(accs))])
cnn_test = predict_cnn(models_, Xt, mean, std, dev)
ens_test = w * cnn_test + (1 - w) * svm_test
res["oof_dev"] = {"svm": metrics(yd, svm_oof), "cnn": metrics(yd, cnn_oof), "ensamble": metrics(yd, w * cnn_oof + (1 - w) * svm_oof)}
res["test_ciego"] = {"svm": metrics(yt, svm_test), "cnn": metrics(yt, cnn_test),
                     "cnn_sin_tta": metrics(yt, predict_cnn(models_, Xt, mean, std, dev, tta=False)),
                     "ensamble": metrics(yt, ens_test)}
if args.extra:
    ex = is_extra; ens_oof = w * cnn_oof + (1 - w) * svm_oof
    res["oof_dominio_extra"] = {"svm": metrics(yd[ex], svm_oof[ex]), "cnn": metrics(yd[ex], cnn_oof[ex]), "ensamble": metrics(yd[ex], ens_oof[ex])}
    print("  OOF en dominio extra:", {k: round(v["acc"], 4) for k, v in res["oof_dominio_extra"].items()})
if args.prof_dir:
    pm = is_prof; ens_oof = w * cnn_oof + (1 - w) * svm_oof
    res["oof_prof"] = {"svm": metrics(yd[pm], svm_oof[pm]), "cnn": metrics(yd[pm], cnn_oof[pm]), "ensamble": metrics(yd[pm], ens_oof[pm])}
    print("  OOF en imágenes (CV honesta):", {k: round(v["acc"], 4) for k, v in res["oof_prof"].items()})
json.dump({"mean": mean, "std": std, "w_cnn": w, "thr": 0.5}, open("models/meta.json", "w"))
json.dump(res, open("results.json", "w"), indent=1)
for k, v in res["test_ciego"].items():
    print(f"  TEST {k:12s} acc={v['acc']:.4f} prec={v['precision']:.4f} rec={v['recall']:.4f} f1={v['f1']:.4f}")
print(f"\nPeso CNN en ensamble = {w:.2f}. Copia a tu portátil: models/, blind_test/, common.py, app.py")
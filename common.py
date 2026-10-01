"""Módulo compartido entre train.py (PC de mesa, GPU) y app.py (portátil, CPU)."""
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from skimage.feature import hog, local_binary_pattern

# ---------------------------------------------------------------- Preprocesamiento
def preprocess(img, blur=True):
    """CLAHE sobre canal L (LAB) para normalizar contraste + suavizado gaussiano leve."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return cv2.GaussianBlur(out, (3, 3), 0) if blur else out

# ---------------------------------------------------------------- Descriptores visuales
def feature_groups(img):
    """Devuelve dict con HOG (forma/bordes), histogramas HSV (color) y LBP (textura)."""
    p = preprocess(img)
    g = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
    f_hog = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    hsv = cv2.cvtColor(p, cv2.COLOR_RGB2HSV)
    ch = []
    for c in range(3):
        h = cv2.calcHist([hsv], [c], None, [16], [0, 180] if c == 0 else [0, 256]).ravel()
        ch.append(h / (h.sum() + 1e-9))
    stats = np.concatenate([p.reshape(-1, 3).mean(0) / 255, p.reshape(-1, 3).std(0) / 255])
    f_color = np.concatenate(ch + [stats])
    lbp = local_binary_pattern(g, 8, 1, "uniform")
    f_lbp = []
    for a in (slice(0, 40), slice(40, 80)):          # rejilla 2x2
        for b in (slice(0, 40), slice(40, 80)):
            h, _ = np.histogram(lbp[a, b], bins=10, range=(0, 10))
            f_lbp.append(h / (h.sum() + 1e-9))
    return {"hog": f_hog, "color": f_color, "lbp": np.concatenate(f_lbp)}

def extract_features(img, groups=("hog", "color", "lbp")):
    d = feature_groups(img)
    return np.concatenate([d[k] for k in groups]).astype(np.float32)

# ---------------------------------------------------------------- SVM calibrado
class SvmProb:
    """Pipeline (escalado+PCA+SVC) + calibración de Platt ajustada sobre decisiones fuera de fold."""
    def __init__(self, pipe, platt): self.pipe, self.platt = pipe, platt
    def predict_proba(self, X):
        p = self.platt.predict_proba(self.pipe.decision_function(X).reshape(-1, 1))[:, 1]
        return np.stack([1 - p, p], 1)

# ---------------------------------------------------------------- Augmentation para SVM (numpy)
def augment_np(img, rng):
    """Zoom 0.6x-2.4x, rotación libre, traslación, saturación y color aleatorios."""
    z = np.exp(rng.uniform(np.log(.6), np.log(2.4)))
    M = cv2.getRotationMatrix2D((40, 40), rng.uniform(0, 360), z); M[:, 2] += rng.uniform(-4, 4, 2)
    o = cv2.warpAffine(img, M, (80, 80), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    g = o.mean(2, keepdims=True); o = g + (o - g) * rng.uniform(.5, 1.6)
    return np.clip(o * rng.uniform(.8, 1.2, (1, 1, 3)), 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- CNN
class ShipNet(nn.Module):
    """4 bloques Conv-BN-ReLU x2 + MaxPool, GAP, Dropout, 1 logit. Entrada en [0,1]."""
    def __init__(self, w=32, p=0.3):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, w), blk(w, 2 * w), blk(2 * w, 4 * w), blk(4 * w, 8 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p), nn.Linear(8 * w, 1))
    def forward(self, x):
        # estandarización POR IMAGEN y canal: robusta a brillo/contraste/tinte de otro sensor
        m, s = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True)
        return self.head(self.f((x - m) / (s + 0.03))).squeeze(1)

def to_tensor(X, mean=None, std=None):      # mean/std se ignoran (compatibilidad)
    return torch.from_numpy(X.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()

def rescale(x, s):
    """s<1 aleja (barco más pequeño), s>1 acerca. Mantiene 80x80 con reflexión en los bordes."""
    if s == 1: return x
    th = torch.tensor([[1 / s, 0, 0], [0, 1 / s, 0]], dtype=x.dtype, device=x.device).expand(len(x), 2, 3)
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)

@torch.no_grad()
def predict_cnn(models, X, mean=None, std=None, device="cpu", tta=True, bs=256, scales=(1.0,)):
    """Promedio sobre modelos (folds), simetrías D4 (TTA) y escalas."""
    out = np.zeros(len(X), np.float32)
    for i in range(0, len(X), bs):
        xb = to_tensor(X[i:i + bs]).to(device); ps = []
        for s in scales:
            xs = rescale(xb, s)
            views = [torch.rot90(v, k, (2, 3)) for v in (xs, xs.flip(3)) for k in range(4)] if tta else [xs]
            ps += [torch.sigmoid(m.to(device).eval()(v)) for m in models for v in views]
        out[i:i + bs] = torch.stack(ps).mean(0).cpu().numpy()
    return out
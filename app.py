"""
UI de evaluación en vivo (corre en CPU, portátil).  Uso:  python app.py
Atajos: B = barco, N = no barco, ← → navegar, Espacio = modo en vivo.
"""
import os, json, csv, glob, time, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np, cv2, joblib, torch
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from common import ShipNet, predict_cnn, extract_features

M = "models"
NAMES = {0: "No barco", 1: "Barco"}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificador de barcos - evaluación en vivo"); self.geometry("1320x780")
        self.meta, self.cnns, self.svm, self.ready = {"w_cnn": .5, "mean": [0] * 3, "std": [1] * 3, "thr": .5}, [], None, False
        try:
            self.meta = json.load(open(f"{M}/meta.json"))
            for f in sorted(glob.glob(f"{M}/cnn_fold*.pt")):
                n = ShipNet(); n.load_state_dict(torch.load(f, map_location="cpu")); self.cnns.append(n.eval())
            self.svm = joblib.load(f"{M}/svm.joblib"); self.ready = True
        except FileNotFoundError:
            self.title("Clasificador de barcos - SIN MODELOS (solo etiquetado)")
            self.after(500, lambda: messagebox.showinfo("Sin modelos", "No se encontró la carpeta models/.\nPuedes cargar imágenes, etiquetarlas y exportar el CSV, pero no inferir."))
        self.paths, self.imgs, self.labels, self.preds, self.order = [], None, {}, {}, []
        self.idx, self.live = 0, False
        self.model_var = tk.StringVar(value="Ensamble"); self.tta = tk.BooleanVar(value=True)
        self.delay = tk.IntVar(value=600); self.auto = tk.BooleanVar(value=True)
        self.build()
        for k, f in (("<b>", 1), ("<n>", 0)): self.bind(k, lambda e, f=f: self.set_label(f))
        self.bind("<Left>", lambda e: self.goto(self.idx - 1)); self.bind("<Right>", lambda e: self.goto(self.idx + 1))
        self.bind("<space>", lambda e: self.toggle_live())

    # ---------------------------------------------------------------- UI
    def build(self):
        left = ttk.Frame(self, padding=8); left.pack(side="left", fill="y")
        ttk.Button(left, text="📁 Cargar carpeta de test", command=self.load_folder).pack(fill="x")
        row = ttk.Frame(left); row.pack(fill="x", pady=2)
        ttk.Button(row, text="Etiquetas CSV", command=self.load_csv).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Etiq. desde nombre", command=self.labels_from_names).pack(side="left", expand=True, fill="x")
        self.img_lbl = tk.Label(left, bg="#222"); self.img_lbl.pack(pady=6)
        self.info = ttk.Label(left, text="—", font=("Segoe UI", 12, "bold")); self.info.pack()
        self.bar = ttk.Progressbar(left, maximum=100, length=320); self.bar.pack(pady=3)
        row = ttk.Frame(left); row.pack(fill="x", pady=4)
        ttk.Button(row, text="🚢 Barco (B)", command=lambda: self.set_label(1)).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="🌊 No barco (N)", command=lambda: self.set_label(0)).pack(side="left", expand=True, fill="x")
        row = ttk.Frame(left); row.pack(fill="x")
        ttk.Button(row, text="◀", width=4, command=lambda: self.goto(self.idx - 1)).pack(side="left")
        ttk.Button(row, text="Inferir actual", command=lambda: (self.ensure_pred(self.idx), self.show())).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="▶", width=4, command=lambda: self.goto(self.idx + 1)).pack(side="left")
        ttk.Button(left, text="⚡ Inferir todas", command=self.infer_all).pack(fill="x", pady=2)
        self.btn_live = ttk.Button(left, text="▶ Modo en vivo (Espacio)", command=self.toggle_live); self.btn_live.pack(fill="x")
        ttk.Label(left, text="Modelo:").pack(anchor="w", pady=(6, 0))
        cb = ttk.Combobox(left, textvariable=self.model_var, values=["Ensamble", "CNN", "SVM (HOG+color+LBP)"], state="readonly")
        cb.pack(fill="x"); cb.bind("<<ComboboxSelected>>", lambda e: self.reset_preds())
        ttk.Checkbutton(left, text="TTA (8 simetrías)", variable=self.tta, command=self.reset_preds).pack(anchor="w")
        ttk.Checkbutton(left, text="Al etiquetar: inferir y avanzar", variable=self.auto).pack(anchor="w")
        ttk.Label(left, text="Pausa en vivo (ms):").pack(anchor="w"); ttk.Scale(left, from_=100, to=2000, variable=self.delay).pack(fill="x")
        ttk.Button(left, text="💾 Exportar resultados CSV", command=self.export).pack(fill="x", pady=4)
        self.lb = tk.Listbox(left, height=10, width=44, font=("Consolas", 9)); self.lb.pack(fill="both", expand=True)
        self.lb.bind("<<ListboxSelect>>", lambda e: self.goto(self.lb.curselection()[0]) if self.lb.curselection() else None)

        right = ttk.Frame(self, padding=8); right.pack(side="left", fill="both", expand=True)
        self.kpi = ttk.Label(right, text="", font=("Segoe UI", 15, "bold")); self.kpi.pack(anchor="w")
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax1, self.ax2 = self.fig.add_subplot(121), self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, right); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.refresh_metrics()

    # ---------------------------------------------------------------- Datos
    def load_folder(self):
        d = filedialog.askdirectory(title="Carpeta con imágenes de test")
        if not d: return
        ps = sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg")))
        if not ps: return messagebox.showwarning("Vacío", "No hay imágenes en la carpeta.")
        self.imgs = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (80, 80), interpolation=cv2.INTER_AREA) for p in ps])
        self.paths, self.labels, self.idx = ps, {}, 0
        self.lb.delete(0, "end"); [self.lb.insert("end", os.path.basename(p)) for p in ps]
        self.reset_preds(); self.show()

    def load_csv(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not f or not self.paths: return
        lab = {r["filename"]: int(r["label"]) for r in csv.DictReader(open(f))}
        for i, p in enumerate(self.paths):
            if os.path.basename(p) in lab: self.labels[i] = lab[os.path.basename(p)]
        self.refresh_metrics(); self.show()

    def labels_from_names(self):
        for i, p in enumerate(self.paths):
            h = os.path.basename(p).split("__")[0]
            if h in ("0", "1"): self.labels[i] = int(h)
        self.refresh_metrics(); self.show()

    # ---------------------------------------------------------------- Inferencia
    def probs(self, imgs):
        m = self.model_var.get(); w = self.meta["w_cnn"]
        if m.startswith("SVM") or m == "Ensamble":
            ps = self.svm.predict_proba(np.stack([extract_features(i) for i in imgs]))[:, 1]
        if m.startswith("SVM"): return ps
        pc = predict_cnn(self.cnns, imgs, self.meta["mean"], self.meta["std"], "cpu", self.tta.get())
        return pc if m == "CNN" else w * pc + (1 - w) * ps

    def ensure_pred(self, i):
        if not self.ready or not len(self.paths) or i in self.preds: return
        t = time.perf_counter(); p = float(self.probs(self.imgs[i:i + 1])[0])
        self.preds[i] = (p, int(p >= self.meta["thr"]), (time.perf_counter() - t) * 1000)
        self.after_eval(i)

    def infer_all(self):
        if not self.ready: return messagebox.showwarning("Sin modelos", "Entrena primero (train.py) para poder inferir.")
        todo = [i for i in range(len(self.paths)) if i not in self.preds]
        for s in range(0, len(todo), 32):
            ch = todo[s:s + 32]; t = time.perf_counter(); ps = self.probs(self.imgs[ch])
            ms = (time.perf_counter() - t) * 1000 / len(ch)
            for i, p in zip(ch, ps): self.preds[i] = (float(p), int(p >= self.meta["thr"]), ms); self.after_eval(i, draw=False)
            self.info.config(text=f"Inferidas {s + len(ch)}/{len(todo)}"); self.update()
        self.refresh_metrics(); self.show()

    def reset_preds(self):
        self.preds, self.order = {}, []
        for i in range(len(self.paths)): self.lb.itemconfig(i, bg="white")
        self.refresh_metrics(); self.show()

    def after_eval(self, i, draw=True):
        if i in self.labels and i in self.preds and i not in self.order: self.order.append(i)
        if i in self.labels and i in self.preds:
            self.lb.itemconfig(i, bg="#c8f7c5" if self.labels[i] == self.preds[i][1] else "#f7c5c5")
        if draw: self.refresh_metrics()

    # ---------------------------------------------------------------- Interacción
    def set_label(self, v):
        if not self.paths: return
        self.labels[self.idx] = v
        if self.auto.get(): self.ensure_pred(self.idx)
        self.after_eval(self.idx); self.show()
        if self.auto.get() and not self.live: self.goto(self.idx + 1)

    def goto(self, i):
        if self.paths and 0 <= i < len(self.paths): self.idx = i; self.show()

    def show(self):
        if not self.paths: return
        i = self.idx; im = Image.fromarray(self.imgs[i]).resize((320, 320), Image.NEAREST)
        self.tk_im = ImageTk.PhotoImage(im); self.img_lbl.config(image=self.tk_im)
        self.lb.selection_clear(0, "end"); self.lb.selection_set(i); self.lb.see(i)
        gt = NAMES.get(self.labels.get(i), "sin etiquetar")
        if i in self.preds:
            p, yp, ms = self.preds[i]
            self.info.config(text=f"[{i + 1}/{len(self.paths)}] Pred: {NAMES[yp]} ({p:.1%})  |  Real: {gt}  |  {ms:.0f} ms")
            self.bar["value"] = p * 100
        else:
            self.info.config(text=f"[{i + 1}/{len(self.paths)}] Real: {gt}  |  sin inferencia"); self.bar["value"] = 0

    # ---------------------------------------------------------------- Modo en vivo
    def toggle_live(self):
        if not self.paths: return
        self.live = not self.live
        self.btn_live.config(text="⏸ Detener (Espacio)" if self.live else "▶ Modo en vivo (Espacio)")
        if self.live: self.live_step()

    def live_step(self):
        if not self.live: return
        self.ensure_pred(self.idx); self.show()
        if self.idx not in self.labels: return self.after(150, self.live_step)   # espera a que etiquetes
        if self.idx >= len(self.paths) - 1: return self.toggle_live()
        self.idx += 1; self.after(int(self.delay.get()), self.live_step)

    # ---------------------------------------------------------------- Métricas
    def refresh_metrics(self):
        ev = [i for i in self.order if i in self.labels and i in self.preds]
        yt = np.array([self.labels[i] for i in ev]); yp = np.array([self.preds[i][1] for i in ev])
        tp = int(((yt == 1) & (yp == 1)).sum()); tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
        n = len(ev); acc = (tp + tn) / n if n else 0
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        self.kpi.config(text=f"N={n}  Acc={acc:.2%}  Precisión={pr:.2%}  Recall={rc:.2%}  F1={f1:.2%}")
        self.ax1.clear(); cm = np.array([[tn, fp], [fn, tp]])
        self.ax1.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm): self.ax1.text(c, r, v, ha="center", va="center", fontsize=16, color="red" if r != c else "black")
        self.ax1.set_xticks([0, 1], ["No barco", "Barco"]); self.ax1.set_yticks([0, 1], ["No barco", "Barco"])
        self.ax1.set_xlabel("Predicho"); self.ax1.set_ylabel("Real"); self.ax1.set_title("Matriz de confusión")
        self.ax2.clear()
        if n:
            ok = (yt == yp).astype(float); self.ax2.plot(np.arange(1, n + 1), np.cumsum(ok) / np.arange(1, n + 1))
        self.ax2.axhline(0.98, color="g", ls="--", label="meta 98%"); self.ax2.set_ylim(0.8, 1.01)
        self.ax2.set_title("Accuracy acumulada en vivo"); self.ax2.set_xlabel("imágenes evaluadas"); self.ax2.legend(loc="lower right")
        self.fig.tight_layout(); self.canvas.draw_idle()

    def export(self):
        f = filedialog.asksaveasfilename(defaultextension=".csv")
        if not f: return
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["filename", "real", "pred", "prob_barco", "ms"])
            for i, p in enumerate(self.paths):
                pr = self.preds.get(i); w.writerow([os.path.basename(p), self.labels.get(i, ""), pr[1] if pr else "", f"{pr[0]:.4f}" if pr else "", f"{pr[2]:.1f}" if pr else ""])

if __name__ == "__main__":
    App().mainloop()